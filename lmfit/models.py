"""Curated model database + the "what fits" engine.

Sizing model (kept simple and conservative):
    vram_gb ≈ params_b × bytes_per_param × ARCH_OVERHEAD + CONTEXT_FLOOR_GB
- ARCH_OVERHEAD (~7%) covers quant block padding, weights vs param count drift,
  and runtime state.
- CONTEXT_FLOOR_GB is a flat budget for KV cache + CUDA/Metal context. It is
  intentionally coarse; `lmfit --context` lets you scale it.

VRAM is the dominant cost, so we treat it as a first-order budget. A model that
"fits" at 21.0 GB on a 22 GB card is reported but flagged as tight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

ARCH_OVERHEAD = 1.07
CONTEXT_FLOOR_GB = 0.6  # KV cache + runtime at a modest context (~8k tokens)

# Default context (tokens) used to scale the KV cache budget.
DEFAULT_CONTEXT_TOKENS = 8192

# Quant tiers, best quality first. bytes/param ≈ measured for the gguf family.
QUANTS = {
    "Q8_0": 1.05,
    "Q6_K": 0.82,
    "Q4_K_M": 0.57,
    "Q3_K_M": 0.42,
    "Q2_K": 0.31,
}
DEFAULT_QUANT = "Q4_K_M"

# Human labels per quant.
QUANT_LABEL = {
    "Q8_0": "near-lossless",
    "Q6_K": "high quality",
    "Q4_K_M": "sweet spot",
    "Q3_K_M": "compact",
    "Q2_K": "minimum",
}


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float  # total parameters, billions (MoE: total, flagged)
    kind: str = "dense"  # "dense" | "moe"
    context: int = DEFAULT_CONTEXT_TOKENS
    family: str = ""
    note: str = ""


# Curated set of widely-used local models. Sizes are public spec numbers.
# Kept deliberately small and real; extend in `extra_models.toml` or via API.
MODELS: List[Model] = [
    Model("Llama 3.2 1B", 1.2, "dense", family="meta"),
    Model("Llama 3.2 3B", 3.2, "dense", family="meta"),
    Model("Phi-4 14B", 14.7, "dense", family="microsoft"),
    Model("Gemma 3 12B", 12.6, "dense", family="google"),
    Model("Mistral Small 3.2 24B", 24.0, "dense", family="mistral"),
    Model("Llama 3.1 8B", 8.03, "dense", family="meta"),
    Model("Llama 3.1 70B", 70.6, "dense", family="meta"),
    Model("Qwen 2.5 72B", 72.7, "dense", family="qwen"),
    Model("Qwen 3 32B", 32.8, "dense", family="qwen"),
    Model("GPT-OSS 20B", 21.0, "dense", family="openai"),
    Model("GPT-OSS 120B", 117.0, "moe", family="openai", note="MoE — ~5B active"),
    Model("DeepSeek-R1 70B", 67.1, "dense", family="deepseek"),
    Model("DeepSeek-R1 0528", 671.0, "moe", family="deepseek", note="MoE — 37B active"),
    Model("GLM-4.5-Air 106B", 106.0, "moe", family="zhipu", note="MoE — 12B active"),
    Model("Gemma 3 27B", 27.0, "dense", family="google"),
    Model("Llama 4 Scout 109B", 109.0, "moe", family="meta", note="MoE — 17B active"),
]

# Fast lookup by (loose) name.
_BY_NAME: Dict[str, Model] = {m.name.lower(): m for m in MODELS}


def vram_for_params(params_b: float, quant: str = DEFAULT_QUANT) -> float:
    """GB of VRAM needed for `params_b` billion parameters at a quant tier."""
    bpp = QUANTS.get(quant, QUANTS[DEFAULT_QUANT])
    return params_b * bpp * ARCH_OVERHEAD + CONTEXT_FLOOR_GB


def quant_order() -> List[str]:
    return list(QUANTS.keys())


@dataclass
class Fit:
    model: Model
    quant: str
    vram_gb: float
    fits_vram: bool
    fits_offload: bool = False
    tight: bool = False
    reason: str = ""


@dataclass
class FitResult:
    vram_gb: Optional[float]
    free_ram_gb: Optional[float]
    total_ram_gb: Optional[float]
    fits: List[Fit] = field(default_factory=list)
    best: Optional[Fit] = None
    skipped: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "vram_gb": self.vram_gb,
            "free_ram_gb": self.free_ram_gb,
            "total_ram_gb": self.total_ram_gb,
            "best": _fit_dict(self.best) if self.best else None,
            "fits": [_fit_dict(f) for f in self.fits],
            "skipped": self.skipped,
        }


def _fit_dict(f: Fit) -> dict:
    return {
        "model": f.model.name,
        "params_b": f.model.params_b,
        "kind": f.model.kind,
        "quant": f.quant,
        "vram_gb": round(f.vram_gb, 2),
        "fits_vram": f.fits_vram,
        "fits_offload": f.fits_offload,
        "tight": f.tight,
        "reason": f.reason,
    }


def compute_fits(
    models: List[Model],
    vram_gb: Optional[float],
    free_ram_gb: Optional[float],
    total_ram_gb: Optional[float],
) -> FitResult:
    """For each model, find the best quant tier that fits (or would fit with
    CPU offload), and flag tight/edge cases.

    A model is:
      - FITS: best quant fits fully in VRAM.
      - OFFLOAD: weights don't fit in VRAM but there is RAM headroom to spill
        (marked, slower).
      - SKIP: can't fit either way.
    """
    res = FitResult(vram_gb=vram_gb, free_ram_gb=free_ram_gb, total_ram_gb=total_ram_gb)
    if not vram_gb:
        res.skipped = [m.name for m in models]
        return res

    for m in sorted(models, key=lambda x: -x.params_b):
        # Pick the best (highest-quality) quant that fits fully in VRAM.
        chosen: Optional[Fit] = None
        for q in quant_order():
            v = vram_for_params(m.params_b, q)
            if v <= vram_gb:
                tight = v > vram_gb * 0.92  # within 8% of the ceiling
                chosen = Fit(
                    model=m, quant=q, vram_gb=v, fits_vram=True, tight=tight,
                    reason=(
                        "tight — near VRAM ceiling, keep context short"
                        if tight
                        else "fits with headroom"
                    ),
                )
                break
        if chosen:
            res.fits.append(chosen)
            continue

        # No full-fit tier. Would CPU offload rescue it?
        # Overflow = min quant VRAM - available VRAM. Need RAM for the spill.
        min_vram = vram_for_params(m.params_b, quant_order()[-1])
        overflow_gb = max(0.0, min_vram - vram_gb)
        if free_ram_gb is not None and free_ram_gb >= overflow_gb + 4.0:
            res.fits.append(
                Fit(
                    model=m, quant=quant_order()[-1], vram_gb=min_vram,
                    fits_vram=False, fits_offload=True,
                    reason=f"needs CPU offload (~{overflow_gb:.0f} GB to RAM) — slower",
                )
            )
        else:
            res.skipped.append(f"{m.name} ({m.params_b:g}B)")

    # Rank: full-fit first (largest params first), then offload fits.
    res.fits.sort(key=lambda f: (not f.fits_vram, -f.model.params_b))
    if res.fits:
        # best = largest that fully fits; else largest offload.
        full = [f for f in res.fits if f.fits_vram]
        res.best = (full or res.fits)[0]
    return res
