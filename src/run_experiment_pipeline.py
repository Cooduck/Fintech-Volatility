"""Run the consolidated historical study from a single command."""

import argparse
import sys


def portfolio_options(parser):
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="report coverage and projected fees")
    modes.add_argument("--offline", action="store_true", help="verify the published nine-stock backtest")
    parser.add_argument("--phase", choices=("a", "b"), default="a")
    parser.add_argument(
        "--approved-usd", type=float,
        help="approved additional Phase B budget, including resumed runs",
    )
    parser.add_argument("--study", choices=("v5",), help=argparse.SUPPRESS)


def collect(argv):
    from collection import (
        cnbc, collect_market_bundle, collect_sec, finnhub, fool, gdelt, sec_earnings, specialist, toms,
    )
    from config import NAMES

    providers = {
        "cnbc": cnbc,
        "fool": fool,
        "gdelt": gdelt,
        "finnhub": finnhub,
        "sec": collect_sec,
        "sec_earnings": sec_earnings,
        "specialist": specialist,
        "toms": toms,
        "market": collect_market_bundle,
    }
    parser = argparse.ArgumentParser(prog=f"{sys.argv[0]} collect", description=collect.__doc__)
    parser.add_argument("provider", choices=providers)
    parser.add_argument("--ticker", action="append", choices=NAMES)
    parser.add_argument("--phase", choices=("a", "b"), default="b")
    args = parser.parse_args(argv)
    if args.provider == "market":
        providers["market"](args.phase)
    elif args.provider == "fool":
        providers["fool"](args.ticker)
    else:
        if args.ticker:
            parser.error("--ticker is supported by the fool collector")
        providers[args.provider]()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["collect"]:
        return collect(argv[1:])
    if argv[:1] == ["clock-diagnostic"]:
        parser = argparse.ArgumentParser(
            prog=f"{sys.argv[0]} clock-diagnostic",
            description="Compare news clocks on identical historical windows.",
        )
        parser.parse_args(argv[1:])
        from clock_diagnostic import main as diagnostic

        return diagnostic()

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Additional commands: collect, clock-diagnostic.",
    )
    portfolio_options(parser)
    args = parser.parse_args(argv)
    if args.offline and (args.phase != "b" or args.approved_usd is not None):
        parser.error("--offline requires --phase b and excludes --approved-usd")
    if args.phase == "a" and args.approved_usd is not None:
        parser.error("--approved-usd applies only to Phase B")
    if args.phase == "b":
        from portfolio import main as run_portfolio

        return run_portfolio(args)
    if args.dry_run:
        print("Phase A uses only archived local data. No API calls.")
        return
    from research import run

    return run()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
