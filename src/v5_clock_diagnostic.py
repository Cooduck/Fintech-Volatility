"""Compare historical availability clocks on the same qualified stocks and days."""

from functools import partial
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from improved_pipeline import DATES
from v3_pipeline import with_target
from v4_models import walk_forward
from v4_pipeline import output_dir
from v5_models import GROUPS, derive, fit_route
from v5_pipeline import OUT


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

    def gains(positions):
        label = truth[positions]
        positive = label.sum()
        negative = len(label) - positive
        if not positive or not negative:
            return None
        ranks = rankdata(scores[positions], axis=0)
        auc = (ranks[label].sum(axis=0) - positive * (positive + 1) / 2) / (
            positive * negative
        )
        first, second = np.diff(auc)[[0, 2]]
        return [first, second, second - first]

    sessions = list(paired.groupby("date").indices.values())
    length = len(sessions)
    rng = np.random.default_rng(20261119)
    samples = []
    for _ in range(1000):
        starts = rng.integers(length, size=(length + 4) // 5)
        draw = ((starts[:, None] + np.arange(5)) % length).ravel()[:length]
        result = gains(np.concatenate([sessions[i] for i in draw]))
        if result is not None:
            samples.append(result)
    point = gains(np.arange(len(paired)))
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    return [
        dict(contrast=name, n=len(paired), positive=int(truth.sum()), estimate=value,
             lo95=lo, hi95=hi, valid_draws=len(samples))
        for name, value, lo, hi in zip(
            ["conservative_D-A", "bounded_D-A", "clock_difference"],
            point, low, high,
        )
    ]


def main():
    out = OUT / "phase_b"
    mixed = output_dir("b")
    archived = pd.read_parquet(out / "regular_predictions.parquet")
    qualified = archived[archived.scope.eq("qualified_only")]
    symbols = sorted(set(qualified[qualified.clock.eq("conservative")].symbol)
                     & set(qualified[qualified.clock.eq("bounded")].symbol))
    assert symbols
    variant = json.loads((OUT / "frozen_manifest.json").read_text(encoding="utf8"))["model_variant"]
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


if __name__ == "__main__":
    main()
