# lmfit

**Stop guessing whether that LLM will run on your card. And stop paying for the
API when you shouldn't be.**

`lmfit` does two things in one command:

1. **What fits** — detect your GPU / VRAM / RAM and tell you the *biggest*
   model in a curated set that actually runs, at the best quant that fits,
   flagging the edge cases ("tight", "needs CPU offload — slower").
2. **Breakeven** — the number that nobody prints: *how many requests a month do
   you need before self-hosting beats the API?* It amortises your hardware and
   counts the electricity, so you get an honest verdict, not fanboi math.

Zero required dependencies. Pure stdlib core; `psutil` and `rich` are optional
niceties. Works on Windows, Linux, and macOS.

---

## Install

```bash
pip install lmfit
```

Or run straight from source, no install needed:

```bash
git clone https://github.com/lmfit/lmfit
cd lmfit
python -m lmfit
```

<details>
<summary>Optional extras</summary>

```bash
pip install lmfit[psutil]   # better RAM detection (esp. Windows/macOS)
pip install lmfit[rich]     # prettier tables when printing to a TTY
```
</details>

---

## Usage

### The default — point it at your machine

`lmfit` auto-detects your GPU, VRAM, and RAM, then gives you the verdict:

```text
==============================================================
lmfit v0.1.0  —  biggest local LLM that fits + breakeven
==============================================================

HARDWARE
  GPU: NVIDIA GeForce RTX 4090 (24.0 GB VRAM)
  RAM: 31.8 GB total, 5.4 GB free
  OS: Windows-10-10.0.26200-SP0 (AMD64)

WHAT FITS  (best quant per model, largest first)
  [OK ] DeepSeek-R1 70B            67.1B  Q2_K    22.9 GB  tight — near VRAM ceiling
  [OK ] Qwen 3 32B                 32.8B  Q4_K_M  20.6 GB  fits with headroom
  [OK ] Gemma 3 27B                27B    Q4_K_M  17.1 GB  fits with headroom
  [OK ] Mistral Small 3.2 24B      24B    Q6_K    21.7 GB  fits with headroom
  [OK ] GPT-OSS 20B                21B    Q6_K    19.0 GB  fits with headroom
  [OK ] Phi-4 14B                  14.7B  Q8_0    17.1 GB  fits with headroom
  [OFF] Qwen 2.5 72B               72.7B  Q2_K    24.7 GB  needs CPU offload — slower
  [OFF] Llama 3.1 70B              70.6B  Q2_K    24.0 GB  needs CPU offload — slower

  => BEST FULL-FIT: DeepSeek-R1 70B @ Q2_K (~22.9 GB VRAM)
     ^ tight on VRAM — keep context short or drop a quant

BREAKEVEN  (self-host vs API)
  API per request   : $0.0105
  Local per request : $0.0007  (fixed $97.22/mo amortised)
  Breakeven         : ~9,950 req/month (~332/day)
  At 30,000 req/month:
      API   : $315.00
      Local : $119.10
      => LOCAL WINS by $195.90/month

==============================================================
Heuristics: ~7% arch overhead + flat 0.6 GB KV/runtime budget.
Sizes are public specs; verify with the model card before committing.
```

That's the whole thing. One command, an honest answer.

### Point it at a *different* machine

Planning to buy a GPU, or checking a friend's box? Target it directly:

```bash
lmfit --vram 12 --ram 32              # "would this fit on a 12 GB / 32 GB box?"
lmfit --vram 8 --ram 16 --requests 5000
```

### Set the breakeven to your real usage

The defaults are a sane starting point. Plug in *your* numbers:

```bash
lmfit \
  --requests 50000 \          # how many requests you actually make / month
  --in-price 3 --out-price 15 \   # $ per 1M in/out tokens (your API)
  --in-tokens 2000 --out-tokens 1500 \  # avg token lengths
  --hw-cost 2800 --lifespan 48 \      # your hardware cost + how long you'll keep it
  --watts 420 --rate 0.38 \          # your power draw + electricity price
  --tps 45                          # realistic generation speed on your card
```

### Scripts & CI

```bash
lmfit --json                       # machine-readable, for pipelines
lmfit --list-models                # the curated model set, sorted
lmfit --no-breakeven               # hardware verdict only
```

---

## How it works

### What fits

For a model of `P` billion parameters at quant `q`, VRAM needed is:

```
vram ≈ P × bytes_per_param(q) × 1.07 + 0.6
```

- `bytes_per_param` — measured size of common gguf quants (`Q8_0` ≈ 1.05,
  `Q4_K_M` ≈ 0.57, …).
- `× 1.07` — architecture overhead (block padding, weight/param drift, runtime).
- `+ 0.6 GB` — flat budget for KV cache + CUDA/Metal context at a modest context.

For each model we pick the **highest-quality quant that still fits** in your
VRAM. If nothing fits fully but you have RAM headroom, we mark it as a
**CPU-offload** candidate (it'll be slower — we say so). If it fits neither, it
skips. MoE models are flagged so you know active params ≠ total params.

> It's a first-order budget, deliberately conservative. The `tight` flag exists
> exactly because a 21.0 GB model on a 22 GB card *might* OOM with long context.

### Breakeven

```
API  / req  = (in_tok · $/M_in + out_tok · $/M_out) / 1e6
Local / req ≈ energy of the generated tokens           (tiny)
Local fixed = hardware_cost / lifespan_months          (amortised)

breakeven req/month = Local fixed / (API/req − Local/req)
```

This is the part people get wrong. The GPU is a **fixed cost you amortise**;
the electricity is marginal and small. So the question is never "is a 4090
faster than the API" (obviously yes) — it's **"at my volume, does the amortised
box pay for itself?"** That's what `lmfit` answers.

---

## Design notes

- **Zero required deps.** Detection uses `nvidia-smi`, `psutil` (optional),
  `sysctl`/`/proc`/Windows `GlobalMemoryStatusEx`. No network calls, no GPU
  driver required at import time.
- **Never crashes on detection.** Bad/missing hardware → a value or `None`, and
  a warning. The CLI degrades to breakeven-only if no GPU is found.
- **Honest by default.** We don't pretend a Q2_K 70B is the same as a Q4_K_M
  70B, and we don't pretend offload is free. The flags are the point.

## Limitations (be honest, they're real)

- Sizing is a **budget**, not a guarantee. Long context, huge batch, or an
  unusually fat runtime can push a "fits" model into OOM. The `tight` flag and
  the disclaimer are there for that.
- The curated set is **opinionated and small** (the models people actually run).
  Add your own in `lmfit/models.py` — it's a flat list of `Model(name, params_b,
  kind, ...)`.
- Breakeven uses **average** request sizes. Variable workloads shift the number.
- No multi-GPU sharding model yet — VRAM is treated as a single budget.
- Energy model is simplified (steady `--watts`), no idle-vs-load distinction.

## Roadmap (want to contribute? these are open)

- [ ] `--multi-gpu N` to model sharded inference across N cards
- [ ] `--context` to scale the KV-cache budget for long-context runs
- [ ] `--models models.toml` to load a custom model set from disk
- [ ] A `lmfit compare` mode: two GPUs side by side (buy which one?)
- [ ] Throughput-aware breakeven (parallel requests, not just volume)
- [ ] `pipx`-friendly single-module build
- [ ] A tiny web dashboard (`lmfit serve`)

## Contributing

Fork → branch → PR. The bar is low on the bar: **no required deps, no network
calls in the core, a test with you**. Tests live in `tests/` and run on
Python 3.9–3.12 across Linux/Windows/macOS (see CI).

```bash
pip install -e .[dev]
pytest
```

## License

MIT — do what you want, just keep the attribution and don't sue us.
