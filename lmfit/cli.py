"""lmfit CLI — detect hardware, recommend the biggest local LLM that fits, and
compute the breakeven where self-hosting beats the API.

Usage:
    lmfit                      # auto-detect this machine, print the verdict
    lmfit --vram 12 --ram 32   # target a different machine (GB)
    lmfit --requests 50000     # breakeven at your real usage level
    lmfit --json               # machine-readable output (CI / scripts)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .hardware import detect
from .models import MODELS, FitResult, compute_fits
from .pricing import (
    ApiCostParams,
    LocalCostParams,
    breakeven,
)


def _has_rich() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except Exception:
        return False


# ---- plain-text rendering (no deps) -----------------------------------------

def _fmt_fit_line(f) -> str:
    tag = "OK " if f.fits_vram else "OFF"
    tight = " (tight)" if f.tight else ""
    kind = "MoE" if f.model.kind == "moe" else ""
    return (
        f"  [{tag}] {f.model.name:<28} {f.model.params_b:>6g}B {kind:<3} "
        f"{f.quant:<6} {f.vram_gb:>6.1f} GB  {f.reason}{tight}"
    )


def _render_text(hw, fitres, be, args) -> str:
    out: List[str] = []
    bar = "=" * 62
    out.append(bar)
    out.append(f"lmfit v{__version__}  —  biggest local LLM that fits + breakeven")
    out.append(bar)
    out.append("")
    out.append("HARDWARE")
    for line in hw.summary().splitlines():
        out.append(f"  {line}")
    out.append("")

    out.append("WHAT FITS  (best quant per model, largest first)")
    if fitres.fits:
        for f in fitres.fits[: args.max_results]:
            out.append(_fmt_fit_line(f))
        best = fitres.best
        if best:
            out.append("")
            out.append(
                f"  => BEST FULL-FIT: {best.model.name} @ {best.quant} "
                f"(~{best.vram_gb:.1f} GB VRAM)"
            )
            if best.tight:
                out.append("     ^ tight on VRAM — keep context short or drop a quant")
    else:
        out.append("  (nothing in the curated set fits this hardware)")
    if fitres.skipped and args.verbose:
        out.append("")
        out.append(f"  skipped: {', '.join(fitres.skipped)}")
    out.append("")

    if be is not None:
        out.append("BREAKEVEN  (self-host vs API)")
        out.append(
            f"  API per request   : ${be.api_cost_per_req:.4f}"
        )
        out.append(
            f"  Local per request : ${be.local_cost_per_req:.4f}  "
            f"(fixed ${be.local_fixed_per_month:.2f}/mo amortised)"
        )
        if be.breakeven_req_per_month is not None:
            out.append(
                f"  Breakeven         : ~{be.breakeven_req_per_month:,.0f} req/month "
                f"(~{be.breakeven_req_per_day:,.0f}/day)"
            )
        out.append(f"  At {be.monthly_requests:,} req/month:")
        out.append(f"      API   : ${be.api_month_cost:,.2f}")
        out.append(f"      Local : ${be.local_month_cost:,.2f}")
        if be.local_cheaper:
            out.append(f"      => LOCAL WINS by ${be.savings_month:,.2f}/month")
        else:
            out.append(
                f"      => API WINS by ${abs(be.savings_month):,.2f}/month "
                "(self-host not economical at this volume)"
            )
        out.append("")
    out.append(bar)
    out.append("Heuristics: ~7% arch overhead + flat 0.6 GB KV/runtime budget.")
    out.append("Sizes are public specs; verify with the model card before committing.")
    return "\n".join(out)


def _render_rich(hw, fitres, be, args) -> None:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(Panel.fit(
        f"lmfit [bold cyan]v{__version__}[/] — biggest local LLM that fits + breakeven",
        border_style="cyan",
    ))

    # Hardware
    hw_table = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    hw_table.add_column(style="bold cyan", no_wrap=True)
    hw_table.add_column()
    for line in hw.summary().splitlines():
        k, _, v = line.partition(":")
        hw_table.add_row(k.strip(), v.strip())
    console.print(hw_table)
    console.print()

    # Fits
    if fitres.fits:
        t = Table(title="What fits (best quant per model)", box=box.ROUNDED,
                  show_lines=False)
        t.add_column("Status", justify="center", no_wrap=True)
        t.add_column("Model", style="bold")
        t.add_column("Params", justify="right")
        t.add_column("Type")
        t.add_column("Quant", style="magenta")
        t.add_column("VRAM", justify="right")
        t.add_column("Note")
        for f in fitres.fits[: args.max_results]:
            if f.fits_vram:
                status = "[green]OK[/]"
            else:
                status = "[yellow]OFF[/]"
            tight = " (tight)" if f.tight else ""
            t.add_row(
                status,
                f.model.name,
                f"{f.model.params_b:g}B",
                "MoE" if f.model.kind == "moe" else "dense",
                f.quant,
                f"{f.vram_gb:.1f} GB",
                f.reason + tight,
            )
        console.print(t)
        best = fitres.best
        if best:
            msg = (
                f"Best full-fit: [bold]{best.model.name}[/] @ {best.quant} "
                f"(~[bold]{best.vram_gb:.1f} GB[/] VRAM)"
            )
            if best.tight:
                msg += "  [yellow]^ tight — keep context short or drop a quant[/]"
            console.print("\n[dim cyan]=>[/dim cyan] " + msg)
    else:
        console.print("[red]Nothing in the curated set fits this hardware.[/red]")

    if be is not None:
        bt = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        bt.add_column(style="dim")
        bt.add_column(justify="right")
        bt.add_row("API cost / request", f"${be.api_cost_per_req:.4f}")
        bt.add_row(
            "Local cost / request",
            f"${be.local_cost_per_req:.4f}  (${be.local_fixed_per_month:.2f}/mo amortised)",
        )
        if be.breakeven_req_per_month is not None:
            bt.add_row(
                "Breakeven",
                f"~{be.breakeven_req_per_month:,.0f} req/month "
                f"({be.breakeven_req_per_day:,.0f}/day)",
            )
        bt.add_row("", "")
        bt.add_row(f"At {be.monthly_requests:,} req/month — API", f"${be.api_month_cost:,.2f}")
        bt.add_row("Local", f"${be.local_month_cost:,.2f}")
        console.print(bt)
        if be.local_cheaper:
            console.print(
                f"[bold green]Local wins by ${be.savings_month:,.2f}/month[/]"
            )
        else:
            console.print(
                f"[bold yellow]API wins by ${abs(be.savings_month):,.2f}/month[/] "
                f"at this volume[/]"
            )

    console.print(
        "\n[dim]Heuristics: ~7% arch overhead + flat 0.6 GB KV/runtime budget. "
        "Sizes are public specs — verify against the model card.[/dim]"
    )


# ---- main -------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="lmfit",
        description="Find the biggest local LLM that fits your hardware, and the "
                    "breakeven where self-hosting beats the API.",
    )
    p.add_argument("--version", action="version", version=f"lmfit {__version__}")
    p.add_argument("--vram", type=float, default=None,
                   help="Target VRAM in GB (default: auto-detect).")
    p.add_argument("--ram", type=float, default=None,
                   help="Target free RAM in GB (default: auto-detect).")
    p.add_argument("--requests", type=int, default=30_000,
                   help="Monthly request volume for the breakeven (default: 30000).")
    # API model pricing
    p.add_argument("--in-price", type=float, default=3.0,
                   help="API $ per 1M input tokens (default: 3.0).")
    p.add_argument("--out-price", type=float, default=15.0,
                   help="API $ per 1M output tokens (default: 15.0).")
    p.add_argument("--in-tokens", type=int, default=1000,
                   help="Avg input tokens/request (default: 1000).")
    p.add_argument("--out-tokens", type=int, default=500,
                   help="Avg output tokens/request (default: 500).")
    # Local cost
    p.add_argument("--hw-cost", type=float, default=3500.0,
                   help="GPU+box hardware cost in $ (default: 3500, set 0 if owned).")
    p.add_argument("--lifespan", type=int, default=36,
                   help="Amortisation window in months (default: 36).")
    p.add_argument("--watts", type=float, default=350.0,
                   help="Sustained system power draw in W (default: 350).")
    p.add_argument("--rate", type=float, default=0.30,
                   help="Electricity $/kWh (default: 0.30).")
    p.add_argument("--tps", type=float, default=60.0,
                   help="Realistic generation speed tokens/sec (default: 60).")
    # Output
    p.add_argument("--json", action="store_true", help="JSON output.")
    p.add_argument("--max-results", type=int, default=12,
                   help="Max model rows to print (default: 12).")
    p.add_argument("--no-breakeven", action="store_true",
                   help="Skip the API breakeven section.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose (list skipped models).")
    p.add_argument("--list-models", action="store_true",
                   help="List the curated model set and exit.")
    args = p.parse_args(argv)

    if args.list_models:
        for m in sorted(MODELS, key=lambda x: -x.params_b):
            kind = "MoE" if m.kind == "moe" else "dense"
            print(f"{m.name:<28} {m.params_b:>6g}B  {kind:<5} {m.family:<10} {m.note}")
        return 0

    hw = detect()
    vram_gb = args.vram if args.vram is not None else hw.primary_vram_gb
    free_ram_gb = args.ram if args.ram is not None else hw.free_ram_gb
    total_ram_gb = hw.total_ram_gb

    fitres = compute_fits(MODELS, vram_gb, free_ram_gb, total_ram_gb)

    be = None
    if not args.no_breakeven:
        local = LocalCostParams(
            hardware_cost=args.hw_cost, lifespan_months=args.lifespan,
            watts=args.watts, electricity_rate=args.rate,
            tokens_per_second=args.tps,
        )
        api = ApiCostParams(
            input_price=args.in_price, output_price=args.out_price,
            avg_input_tokens=args.in_tokens, avg_output_tokens=args.out_tokens,
        )
        be = breakeven(local, api, monthly_requests=args.requests)

    if args.json:
        payload = {
            "version": __version__,
            "hardware": {
                "gpus": [{"name": g.name, "vram_gb": g.vram_gb, "kind": g.kind}
                         for g in hw.gpus],
                "total_ram_gb": total_ram_gb,
                "free_ram_gb": free_ram_gb,
                "os": hw.os,
                "arch": hw.arch,
            },
            "fit": fitres.as_dict(),
            "breakeven": be.as_dict() if be else None,
        }
        print(json.dumps(payload, indent=2))
        return 0

    if vram_gb is None:
        print("lmfit: no GPU/VRAM detected. Use --vram <GB> to target a machine.")
        sys.stderr.write("Running breakeven-only mode.\n\n")

    if _has_rich() and sys.stdout.isatty():
        _render_rich(hw, fitres, be, args)
    else:
        print(_render_text(hw, fitres, be, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
