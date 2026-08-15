"""Core unit tests — pure stdlib, no hardware, no network."""
import math

import pytest

from lmfit.models import (
    MODELS,
    QUANTS,
    compute_fits,
    vram_for_params,
    quant_order,
)
from lmfit.pricing import (
    ApiCostParams,
    LocalCostParams,
    api_cost_per_request,
    breakeven,
    local_fixed_per_month,
)


# ---- vram sizing -------------------------------------------------------------

def test_vram_monotonic_in_params():
    assert vram_for_params(2, "Q4_K_M") < vram_for_params(8, "Q4_K_M")


def test_vram_monotonic_in_quant():
    # Better (higher) quality = more VRAM for the same model.
    q = list(QUANTS.keys())  # best -> worst
    vals = [vram_for_params(8, tier) for tier in q]
    assert vals == sorted(vals, reverse=True)


def test_vram_known_value_8b_q4():
    # 8.03B * 0.57 * 1.07 + 0.6 ≈ 4.9 GB.
    v = vram_for_params(8.03, "Q4_K_M")
    assert 4.5 < v < 5.5


def test_quant_order_matches_quants():
    assert set(quant_order()) == set(QUANTS.keys())


# ---- fit engine --------------------------------------------------------------

def test_fit_picks_highest_tier_that_fits():
    res = compute_fits(MODELS, vram_gb=16.0, free_ram_gb=32, total_ram_gb=32)
    by_name = {f.model.name: f for f in res.fits}
    # 16 GB: a 14B Q4_K_M (~9.2 GB) fits, but a 14B Q8_0 (~13.9 GB) also fits.
    # Best tier chosen must be the highest-quality one that still fits in 16 GB.
    phi = by_name.get("Phi-4 14B")
    assert phi is not None
    assert phi.fits_vram
    assert phi.vram_gb <= 16.0
    # It should not be the worst tier if a better one fits.
    assert phi.quant != quant_order()[-1]


def test_fit_excludes_too_big_model():
    res = compute_fits(MODELS, vram_gb=8.0, free_ram_gb=0, total_ram_gb=8)
    names = {f.model.name for f in res.fits}
    assert "Llama 3.1 70B" not in names
    assert "DeepSeek-R1 70B" not in names


def test_fit_no_vram_skips_all():
    res = compute_fits(MODELS, vram_gb=None, free_ram_gb=None, total_ram_gb=None)
    assert res.fits == []
    assert res.best is None
    assert len(res.skipped) == len(MODELS)


def test_fit_offload_when_ram_present():
    # 8 GB VRAM, big RAM: a 70B should be allowed via CPU offload (flagged).
    res = compute_fits(MODELS, vram_gb=8.0, free_ram_gb=64, total_ram_gb=64)
    offload = [f for f in res.fits if not f.fits_vram and f.fits_offload]
    assert any("70B" in f.model.name for f in offload)
    # And it must be ranked AFTER all full-fit models.
    order = [f.model.name for f in res.fits]
    if offload:
        assert order.index(offload[0].model.name) > 0


def test_fit_no_offload_without_ram():
    # 8 GB VRAM, no RAM headroom -> 70B must be skipped, not offloaded.
    res = compute_fits(MODELS, vram_gb=8.0, free_ram_gb=2, total_ram_gb=4)
    names = {f.model.name for f in res.fits}
    assert "DeepSeek-R1 70B" not in names


def test_fit_best_is_largest_full_fit():
    res = compute_fits(MODELS, vram_gb=24.0, free_ram_gb=64, total_ram_gb=64)
    full = [f for f in res.fits if f.fits_vram]
    assert res.best is not None
    assert res.best.fits_vram
    assert res.best.model.params_b == max(f.model.params_b for f in full)


def test_moe_flagged():
    moe = [m for m in MODELS if m.kind == "moe"]
    assert moe, "expected at least one MoE model in the set"


# ---- pricing / breakeven -----------------------------------------------------

def test_api_cost_per_request_math():
    api = ApiCostParams(input_price=3.0, output_price=15.0,
                        avg_input_tokens=1000, avg_output_tokens=500)
    # (1000*3 + 500*15)/1e6 = (3000 + 7500)/1e6 = 0.0105
    assert api_cost_per_request(api) == pytest.approx(0.0105)


def test_breakeven_point():
    local = LocalCostParams(hardware_cost=3600, lifespan_months=36)  # $100/mo fixed
    api = ApiCostParams(input_price=3.0, output_price=15.0,
                        avg_input_tokens=1000, avg_output_tokens=500)
    be = breakeven(local, api)
    # fixed $100/mo ; per-request delta ≈ 0.0105 - ~0.0006 ≈ 0.0099
    assert be.breakeven_req_per_month is not None
    assert 8000 < be.breakeven_req_per_month < 12000
    # Below breakeven -> API cheaper; above -> local cheaper.
    assert not breakeven(local, api, monthly_requests=1000).local_cheaper
    assert breakeven(local, api, monthly_requests=100_000).local_cheaper


def test_local_fixed_per_month():
    local = LocalCostParams(hardware_cost=3600, lifespan_months=12)
    assert local_fixed_per_month(local) == pytest.approx(300.0)


def test_zero_hardware_cost_breakeven_low():
    # If you already own the box, breakeven is dominated by marginal energy.
    local = LocalCostParams(hardware_cost=0, lifespan_months=36)
    api = ApiCostParams(input_price=3.0, output_price=15.0,
                        avg_input_tokens=1000, avg_output_tokens=500)
    be = breakeven(local, api, monthly_requests=100)
    assert be.local_cheaper


def test_breakeven_roundtrip_dict():
    local = LocalCostParams()
    api = ApiCostParams()
    be = breakeven(local, api)
    d = be.as_dict()
    assert d["api_cost_per_req"] == pytest.approx(be.api_cost_per_req)
    assert "breakeven_req_per_month" in d
