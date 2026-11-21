"""Compare historical availability clocks on the same qualified stocks and days."""

from functools import partial
import json

import pandas as pd

from backtest import output_dir
from config import DATES, FINAL_RESULTS, PORTFOLIO_RESULTS
from evaluation import auc_contrasts
from features import derive, with_target
from models import GROUPS, fit_route, walk_forward


def paired_clock_effect(conservative, bounded):
    columns = ["date", "symbol", "y", "fold", "cut"]
    paired = conservative[columns + ["A", "D"]].merge(
        bounded[columns + ["A", "D"]],
        on=columns,
        suffixes=("_conservative", "_bounded"),
        validate="one_to_one",
    ).sort_values(["date", "symbol"]).reset_index(drop=True)
    assert len(paired) == len(conservative) == len(bounded)
    truth = paired.y.to_numpy().astype(bool)
    scores = paired[
        ["A_conservative", "D_conservative", "A_bounded", "D_bounded"]
    ].to_numpy()

    def gains(auc):
        conservative_gain = auc[1] - auc[0]
        bounded_gain = auc[3] - auc[2]
        return [conservative_gain, bounded_gain, bounded_gain - conservative_gain]

    sessions = list(paired.groupby("date").indices.values())
    point, low, high, valid_draws = auc_contrasts(truth, scores, sessions, gains)
    if point is None:
        raise ValueError("The clock comparison requires both outcome classes.")
    return [
        dict(contrast=name, n=len(paired), positive=int(truth.sum()), estimate=value,
             lo95=lo, hi95=hi, valid_draws=valid_draws)
        for name, value, lo, hi in zip(
            ["conservative_D-A", "bounded_D-A", "clock_difference"],
            point, low, high,
        )
    ]


def main():
    out = PORTFOLIO_RESULTS
    mixed = output_dir("b")
    archived = pd.read_parquet(out / "regular_predictions.parquet")
    qualified = archived[archived.scope.eq("qualified_only")]
    symbols = sorted(set(qualified[qualified.clock.eq("conservative")].symbol)
                     & set(qualified[qualified.clock.eq("bounded")].symbol))
    assert symbols
    variant = json.loads((FINAL_RESULTS / "frozen_manifest.json").read_text(encoding="utf8"))["model_variant"]
    groups = {
        name: [field for field in fields if field != "is_earnings_window"]
        for name, fields in GROUPS.items()
    }
    panel = derive(pd.concat([
        pd.read_parquet(mixed / f"conservative_{symbol}_features.parquet")
        for symbol in symbols
    ], ignore_index=True))
    panel = panel[panel.is_earnings_window.eq(0)].sort_values(["cut", "symbol"])
    keys = ["date", "symbol", "y", "fold", "cut"]
    rows = []
    for target in ("B1", "B2"):
        comparison, _ = walk_forward(
            with_target(panel, target), DATES, target, groups=groups,
            route_fitter=partial(fit_route, candidate=variant),
        )
        bounded = qualified[
            qualified.target.eq(target)
            & qualified.clock.eq("bounded")
            & qualified.symbol.isin(symbols)
        ]
        assert set(qualified[qualified.clock.eq("bounded")].symbol) == set(symbols)
        comparison = comparison.pivot(index=keys, columns="group", values="score").reset_index()
        bounded = bounded.pivot(index=keys, columns="group", values="score").reset_index()
        rows.extend(dict(target=target, symbols="|".join(symbols), **row)
                    for row in paired_clock_effect(comparison, bounded))
    data = pd.DataFrame(rows)
    data.to_csv(out / "clock_matched_diagnostic.csv", index=False)
    print(data.to_string(index=False), flush=True)
