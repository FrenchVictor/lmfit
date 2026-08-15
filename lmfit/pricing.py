"""Breakeven: where self-hosting stops losing money versus the API.

Model (kept transparent and overridable):
  API cost / request  = (in_tok · $/M_in + out_tok · $/M_out) / 1e6
  Local fixed / month = hardware_cost / lifespan_months
  Local marginal / req≈ energy cost of the generated tokens (tiny)
  breakeven req/month = local_fixed_month / (api_per_req - local_per_req)

It answers the real question: "at MY usage, is the 4090 cheaper than the API?"
All inputs have sensible defaults and are overridable from the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LocalCostParams:
    hardware_cost: float = 3500.0      # $ — GPU + box you'd buy (0 if already owned)
    lifespan_months: int = 36          # amortisation window
    watts: float = 350.0               # sustained system draw while generating
    electricity_rate: float = 0.30     # $/kWh (EU average ~0.30)
    tokens_per_second: float = 60.0    # realistic generation speed on your HW
    hours_per_day: float = 8.0         # daily generation time (for energy view)

    def energy_per_hour(self) -> float:
        return self.watts * self.electricity_rate / 1000.0


@dataclass
class ApiCostParams:
    input_price: float = 3.0           # $ per 1M input tokens
    output_price: float = 15.0         # $ per 1M output tokens
    avg_input_tokens: int = 1000
    avg_output_tokens: int = 500


@dataclass
class Breakeven:
    api_cost_per_req: float
    local_cost_per_req: float
    local_fixed_per_month: float
    breakeven_req_per_month: Optional[float]  # None => local never cheaper
    breakeven_req_per_day: Optional[float]
    # Verdict for a given usage level:
    monthly_requests: int
    api_month_cost: float
    local_month_cost: float
    local_cheaper: bool
    savings_month: float  # + local saves, - local costs more

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("api_cost_per_req", "local_cost_per_req", "local_fixed_per_month",
                  "breakeven_req_per_month", "breakeven_req_per_day",
                  "api_month_cost", "local_month_cost", "savings_month"):
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 4)
        return d


def api_cost_per_request(api: ApiCostParams) -> float:
    return (api.avg_input_tokens * api.input_price
            + api.avg_output_tokens * api.output_price) / 1_000_000.0


def local_marginal_cost_per_request(local: LocalCostParams, api: ApiCostParams) -> float:
    """Energy cost of generating one request's worth of tokens (amortisation-free)."""
    if local.tokens_per_second <= 0:
        return 0.0
    # $ per second of generation, then per token.
    energy_per_second = local.watts * local.electricity_rate / 1000.0 / 3600.0
    energy_per_token = energy_per_second / local.tokens_per_second
    tokens = api.avg_input_tokens + api.avg_output_tokens
    return energy_per_token * tokens


def local_fixed_per_month(local: LocalCostParams) -> float:
    return local.hardware_cost / max(1, local.lifespan_months)


def breakeven(
    local: LocalCostParams,
    api: ApiCostParams,
    monthly_requests: int = 30_000,
) -> Breakeven:
    api_per_req = api_cost_per_request(api)
    local_per_req = local_marginal_cost_per_request(local, api)
    fixed = local_fixed_per_month(local)

    if api_per_req <= local_per_req:
        be_month = be_day = None
    else:
        be_month = fixed / (api_per_req - local_per_req)
        be_day = be_month / 30.0

    api_month = api_per_req * monthly_requests
    local_month = fixed + local_per_req * monthly_requests
    cheaper = local_month < api_month

    return Breakeven(
        api_cost_per_req=api_per_req,
        local_cost_per_req=local_per_req,
        local_fixed_per_month=fixed,
        breakeven_req_per_month=be_month,
        breakeven_req_per_day=be_day,
        monthly_requests=monthly_requests,
        api_month_cost=api_month,
        local_month_cost=local_month,
        local_cheaper=cheaper,
        savings_month=api_month - local_month,
    )
