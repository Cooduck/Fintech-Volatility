"""Coverage-gated nine-stock scoring and a separate regular-day decision policy."""

from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analysis_models import folds
from improved_pipeline import DATES
from panels import price_panel
from semantic import ANCHORED, CACHE, key, peak_time, read
from v3_features import timestamp_news
from v3_pipeline import INDEX, with_target
from v4_collect import DATA, NAMES, ROOT
from v4_features import add_market_context, price_reader
from v4_models import walk_forward
from v4_pipeline import digest, output_dir, report_tables, run as mixed_run
from v4_scoring import LEAD_PROMPT_V5, REQUIRED_COVERAGE, estimate, phase_b, prepare
from v5_models import GROUPS, derive, fit_route
from v5_pipeline import OUT


def stage1_quality_bounds(symbol, headlines, mornings, scores):
    ordered = headlines.sort_values("utc")
    cached = [scores.get(key(title, ANCHORED)) for title in ordered.title]
    known = np.array([value is not None for value in cached], dtype=bool)
    negative = np.array(
        [value is not None and value["sentiment_score"] < -0.2 for value in cached],
        dtype=bool,
    )
    material = np.array(
        [
            value is not None
            and value["impact_level"] >= 2
            and value["event_type"] != "market_noise"
            for value in cached
        ],
        dtype=bool,
    )

    def observation(metric, fold, sample, matches, percent):
        count = int(sample.sum())
        scored = int(known[sample].sum())
        hits = int(matches[sample].sum())
        required = (count * percent + 99) // 100
        missing = count - scored
        return dict(
            symbol=symbol, metric=metric, fold=fold, headcount=count, cached=scored,
            uncached=missing, cached_positive=hits, min_required=required,
            additionally_required=max(required - hits, 0),
            possible_from_stage1=bool(count and hits + missing >= required),
            guaranteed_from_stage1=bool(count and hits >= required),
        )

    rows = [
        observation("negative_all_stage1", 0, np.ones(len(ordered), dtype=bool), negative, 15)
    ]
    for fold, dates in enumerate(folds(DATES, True), 1):
        seen = np.zeros(len(ordered), dtype=bool)
        test_start = mornings.loc[mornings.date.isin(dates), "cut"].min()
        train = mornings[mornings.date.lt(min(dates)) & mornings.end.le(test_start)]
        for window in train.itertuples():
            lo, hi = ordered.utc.searchsorted([window.start, window.cut], side="right")
            seen[lo:hi] = True
        rows.append(observation("training_material_stage1", fold, seen, material, 10))
    return rows


def verified_coverage(news):
    columns = ["start", "cut"]
    reference = pd.read_csv(
        ROOT / "results" / "improved" / "finance_NVDA_halfday.csv",
        usecols=columns,
        parse_dates=columns,
    )[columns]
    oos_dates = {day for fold in folds(DATES, True) for day in fold}
    vix = pd.read_csv(DATA / "VIX.csv")
    notices = pd.read_csv(DATA / "earnings_notices.csv")
    scores = {row["key"]: row for row in read("deepseek_cache.jsonl")}
    rows = []
    capacity = []
    quality = []
    for symbol in NAMES:
        panel = price_panel("halfday", symbol, continuous=True, read_bars=price_reader)
        windows = panel[panel.date.isin(DATES)].dropna(subset=["z", "z_intraday"])
        headlines = news.loc[news.symbol.eq(symbol)]
        times = headlines.utc.sort_values()
        counts = times.searchsorted(windows.cut, side="right") - times.searchsorted(
            windows.start, side="right"
        )
        covered = int((counts > 0).sum())
        coverage = covered / len(windows) if len(windows) else 0.0
        rows.append(
            dict(
                symbol=symbol,
                windows=len(windows),
                covered=covered,
                coverage=coverage,
                same_windows_as_reference=(
                    len(windows) == len(reference)
                    and windows[columns].reset_index(drop=True).eq(reference).to_numpy().all()
                ),
            )
        )
        mornings = windows[windows.session.eq("AM")]
        quality.extend(stage1_quality_bounds(symbol, headlines, mornings, scores))
        mornings = mornings[mornings.date.isin(oos_dates)]
        filings = pd.read_csv(DATA / f"{symbol}_earnings.csv")
        labeled = (
            mornings.assign(is_earnings_window=0)
            if mornings.empty
            else add_market_context(mornings, symbol, vix, filings, notices)
        )
        for scope, sample in [
            ("all", labeled),
            ("regular", labeled[labeled.is_earnings_window.eq(0)]),
        ]:
            capacity.append(
                dict(
                    symbol=symbol,
                    scope=scope,
                    news_coverage=coverage,
                    mornings=len(sample),
                    B1=int(sample.z.gt(1.5).sum()),
                    B2=int(sample.z_intraday.gt(1.5).sum()),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(capacity), pd.DataFrame(quality)


def earnings_reviews(index, panel, test_dates, clock):
    reviews = []
    samples = panel[panel.is_earnings_window.eq(1) & panel.date.isin(test_dates)]
    for symbol, windows in samples.groupby("symbol"):
        feed = timestamp_news(index[index.symbol.eq(symbol)], clock)
        refined = feed[
            feed.event_type.eq("earnings")
            & feed.catalyst_certainty.isin(["confirmed", "speculative"])
        ]
        for row in windows.itertuples(index=False):
            lo, hi = refined["asof"].searchsorted([row.start, row.cut], side="right")
            known = refined.iloc[lo:hi]
            guidance = known.loc[
                known.catalyst_certainty.eq("confirmed"), "guidance_direction"
            ]
            directions = set(guidance.dropna()) - {"not_stated", ""}
            direction = "mixed" if len(directions) > 1 else next(iter(directions), "not_stated")
            reviews.append(
                dict(
                    clock=clock,
                    date=row.date,
                    symbol=symbol,
                    cut=row.cut,
                    decision="mandatory_review",
                    lead_count=len(known),
                    guidance_direction=direction,
                    sentiment=known.sentiment_score.mean(),
                    volatility_risk_level=known.impact_level.max(),
                    certainty=" | ".join(known.catalyst_certainty.unique()),
                    rationale=" | ".join(known.rationale.dropna().unique()),
                )
            )
    return reviews


def separate_policy(index, out, variant, *, panel_dir=None):
    panel_dir = output_dir("b") if panel_dir is None else panel_dir
    groups = {
        name: [field for field in fields if field != "is_earnings_window"]
        for name, fields in GROUPS.items()
    }
    parts = []
    earnings = []
    eligible = pd.read_csv(panel_dir / "corpus_quality.csv")
    ready = eligible.assign(
        ok=eligible.coverage.ge(0.70)
        & eligible.negative_all.ge(0.15)
        & eligible.training_material.ge(0.10)
    )
    test_dates = set(
        pd.read_parquet(panel_dir / "predictions.parquet", columns=["date"]).date
    )
    route_fitter = partial(fit_route, candidate=variant)
    for clock in ["conservative", "bounded"]:
        frames = [
            pd.read_parquet(panel_dir / f"{clock}_{symbol}_features.parquet")
            for symbol in index.symbol.unique()
        ]
        panel = derive(pd.concat(frames, ignore_index=True)).sort_values(["cut", "symbol"])
        qualified = ready[ready.clock.eq(clock)].groupby("symbol").ok.all()
        regular = panel[panel.is_earnings_window.eq(0)]
        scopes = {"full_9": regular}
        if qualified.any():
            scopes["qualified_only"] = regular[regular.symbol.isin(qualified.index[qualified])]
        for scope, frame in scopes.items():
            for target in ["B1", "B2"]:
                test, _ = walk_forward(
                    with_target(frame, target), DATES, target,
                    groups=groups, route_fitter=route_fitter,
                )
                parts.append(test.assign(clock=clock, scope=scope, target=target))
        earnings.extend(earnings_reviews(index, panel, test_dates, clock))
    forecasts = pd.concat(parts, ignore_index=True)
    assert forecasts.is_earnings_window.eq(0).all()
    assert forecasts.last_train_end.le(forecasts.cut).all()
    forecasts.to_parquet(out / "regular_predictions.parquet", index=False)
    report_tables(forecasts, out)
    pd.DataFrame(earnings).to_csv(out / "earnings_policy.csv", index=False)


def write_acceptance(out, mixed):
    mixed_ablation = pd.read_csv(mixed / "ablation.csv")
    ci = pd.read_csv(out / "paired_auc.csv")
    granger = pd.read_csv(mixed / "pooled_granger.csv")
    primary = granger[granger.family.eq("primary_preregistered")]
    assert len(primary) == 4
    mixed_qualified = mixed_ablation.query(
        "clock == 'conservative' and scope == 'qualified_only' and group == 'D'"
    )
    regular = ci.query(
        "clock == 'conservative' and slice == 'regular' and contrast == 'D-A'"
    )
    rows = []
    for target in ["B1", "B2"]:
        counts = mixed_qualified.loc[mixed_qualified.target.eq(target), "positive"]
        count = int(counts.iloc[0]) if not counts.empty else 0
        for scope in ["full_9", "qualified_only"]:
            band = regular[regular.target.eq(target) & regular.scope.eq(scope)]
            lower = band.lo95.iloc[0] if not band.empty else float("nan")
            rows.append(
                dict(
                    target=target,
                    regular_scope=scope,
                    qualified_oos_abnormal=count,
                    qualified_count_ge_180=count >= 180,
                    regular_delta_auc=band.delta_auc.iloc[0] if not band.empty else float("nan"),
                    regular_lo95=lower,
                    regular_hi95=band.hi95.iloc[0] if not band.empty else float("nan"),
                    regular_positive_95ci=bool(lower > 0),
                    nine_stock_primary_tests_below_05=int(primary.q_BH.lt(0.05).sum()),
                )
            )
    pd.DataFrame(rows).to_csv(out / "acceptance.csv", index=False)


def offline_reproduction():
    mixed = output_dir("b")
    out = OUT / "phase_b"
    scored = mixed / "scored_news.csv.gz"
    snapshot = json.loads((mixed / "frozen_manifest.json").read_text(encoding="utf8"))
    data = {
        name: expected for name, expected in snapshot["data_and_code"].items()
        if name.startswith(("data/", "results/")) or name == "requirements.txt"
    }
    scored_name = scored.relative_to(ROOT).as_posix()
    if scored_name not in data:
        raise ValueError("The Phase B manifest does not cover the scored news.")
    for name, expected in data.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Phase B input differs from its archived hash: {name}")

    ledger = CACHE / "usage.jsonl"
    before = digest(ledger)
    index = pd.read_csv(scored, low_memory=False)
    candidate = pd.read_csv(OUT / "comparison.csv", nrows=1).candidate.iloc[0]
    common = ["ablation.csv", "earnings_split.csv", "paired_auc.csv"]
    checks = [
        (mixed, [*common, "predictions.parquet", "corpus_quality.csv",
                 "pooled_granger.csv", "event_replay.csv"]),
        (out, [*common, "regular_predictions.parquet", "earnings_policy.csv", "acceptance.csv"]),
    ]
    logs = ROOT / ".log"
    logs.mkdir(exist_ok=True)
    checked = []
    with TemporaryDirectory(prefix="v5-offline-", dir=logs) as directory:
        temp = Path(directory).resolve()
        if not temp.is_relative_to(logs.resolve()):
            raise ValueError("Offline reproduction must use the project log directory.")
        mixed_copy, regular_copy = temp / "mixed", temp / "regular"
        regular_copy.mkdir()
        print(f"Offline Phase B: {len(index):,} archived scores", flush=True)
        mixed_run("b", index, out=mixed_copy, freeze_models=False)
        separate_policy(
            index, regular_copy, candidate if candidate != "v4_hard" else "soft",
            panel_dir=mixed_copy,
        )
        write_acceptance(regular_copy, mixed_copy)
        for (original, names), reproduction in zip(checks, (mixed_copy, regular_copy)):
            for name in names:
                load = pd.read_parquet if name.endswith(".parquet") else pd.read_csv
                pd.testing.assert_frame_equal(
                    load(reproduction / name), load(original / name), rtol=1e-9, atol=1e-12
                )
                checked.append((original / name).relative_to(ROOT).as_posix())

    after = digest(ledger)
    if after != before:
        raise RuntimeError("The API usage ledger changed during offline reproduction.")
    audit = dict(
        archived_news_rows=len(index),
        checked_input_files=len(data),
        scored_news_sha256=data[scored_name],
        usage_before=before,
        usage_after=after,
        reproduced_files=checked,
    )
    (out / "offline_reproduction.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    print(f"Reproduced {len(checked)} historical files. API ledger unchanged.", flush=True)


def main(args):
    if args.offline:
        return offline_reproduction()
    base = pd.read_csv(INDEX)
    headlines = prepare(base)
    forecast = estimate(headlines, lead_prompt=LEAD_PROMPT_V5)
    out = OUT / "phase_b"
    out.mkdir(parents=True, exist_ok=True)
    coverage, capacity, quality = verified_coverage(headlines)
    coverage.to_csv(out / "coverage_verified.csv", index=False)
    capacity.to_csv(out / "oos_label_capacity.csv", index=False)
    quality.to_csv(out / "stage1_quality_bounds.csv", index=False)
    passing = set(coverage.loc[coverage.coverage.ge(0.70), "symbol"])
    forecast.update(
        verified_halfday_coverage=coverage.set_index("symbol").coverage.to_dict(),
        coverage_gate_passed=REQUIRED_COVERAGE.issubset(passing),
        coverage_passing_tickers=sorted(passing),
        required_tickers_below_gate=sorted(REQUIRED_COVERAGE - passing),
    )
    (out / "estimate.json").write_text(json.dumps(forecast, indent=2), encoding="utf8")
    print(json.dumps(forecast, indent=2))
    if args.dry_run:
        return
    if not forecast["coverage_gate_passed"]:
        raise SystemExit("Seven required tickers must reach 70% half-day coverage before paid scoring.")
    price_window = "peak_usd" if peak_time(datetime.now(timezone.utc)) else "off_peak_usd"
    if forecast[price_window] + forecast["accounted_before_usd"] > forecast["project_ceiling_usd"]:
        raise SystemExit("Estimated project cost exceeds the cost ceiling.")
    index = phase_b(base, False, args.approved_usd, lead_prompt=LEAD_PROMPT_V5)
    mixed_run("b", index)
    manifest = json.loads((OUT / "frozen_manifest.json").read_text(encoding="utf8"))
    candidate = manifest["model_variant"]
    separate_policy(index, out, candidate if candidate != "v4_hard" else "soft")
    write_acceptance(out, output_dir("b"))
