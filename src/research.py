"""Compare candidates and retain the selected historical model."""

from functools import partial
import json

import pandas as pd

from config import BENCHMARK_RESULTS, CACHE, CORE, DATES, FINAL_RESULTS, FREEZE, NEWS_INDEX, ROOT
from evaluation import qualified_symbols
from features import derive, inputs, market_frame, timestamp_news, with_target
from market_data import price_panel
from models import GROUPS, HARD_GROUPS, fit_hard_route, fit_route, walk_forward, weights
from artifacts import digest, publish_bundle, read_manifest
from backtest import (
    dashboard,
    granger_family,
    replay_case,
    report_tables,
)

CANDIDATES = ("soft", "signed")


def make_panels():
    panels = {}
    for clock in ["conservative", "bounded"]:
        frames = [
            pd.read_parquet(BENCHMARK_RESULTS / f"{clock}_{symbol}_features.parquet")
            for symbol in CORE
        ]
        panels[clock] = pd.concat(frames, ignore_index=True).sort_values(
            ["cut", "symbol"]
        )
    return panels


def collect_candidate(candidate, panels, raw, news, context):
    preds = []
    cases = []
    coefficients = []
    errors = []
    quality = pd.read_csv(BENCHMARK_RESULTS / "corpus_quality.csv")
    for clock, panel in panels.items():
        data = derive(panel)
        eligible = qualified_symbols(quality, clock)
        samples = {"pooled_3": data, "qualified_only": data[data.symbol.isin(eligible)]}
        samples.update({s: data[data.symbol.eq(s)] for s in CORE})
        for scope, sample in samples.items():
            for target in ["B1", "B2"]:

                def audit(route, fold, group):
                    coefficients.extend(
                        dict(row, scope=scope, clock=clock, target=target)
                        for row in weights(route, fold, group)
                    )

                def replay(route):
                    cases.extend(replay_case(route, raw, news, context, derive))

                callback = (
                    replay if clock == "conservative" and scope == "NVDA" else None
                )
                pred, err = walk_forward(
                    with_target(sample, target),
                    DATES,
                    target,
                    callback,
                    groups=GROUPS,
                    route_fitter=partial(fit_route, candidate=candidate),
                    audit=audit,
                )
                errors.append(err)
                preds.append(pred.assign(clock=clock, scope=scope, target=target))
        print(candidate, clock, "complete", flush=True)
    return (
        pd.concat(preds, ignore_index=True),
        pd.DataFrame(cases),
        pd.DataFrame(coefficients),
        max(errors),
    )


def criterion(name, splits, intervals, replay, granger):
    tab = splits.query(
        "clock=='conservative' and scope=='pooled_3' and slice=='regular'"
    )
    d = tab[tab.group.eq("D")].set_index("target")
    a = tab[tab.group.eq("A")].set_index("target")
    ci = intervals.query(
        "clock=='conservative' and scope=='pooled_3' and slice=='regular' and contrast=='D-A'"
    ).set_index("target")
    replay_ok = {}
    for target, block in replay.groupby("target"):
        timing = block.sort_values("cut").score.iloc[:3]
        replay_ok[target] = bool(timing.diff().iloc[1:].gt(0).all())
    b2 = granger.query(
        "family=='primary_preregistered' and target=='B2' and feature=='ClusterImpactSum'"
    )
    return dict(
        candidate=name,
        b1_strict=replay_ok["B1"],
        b2_strict=replay_ok["B2"],
        strict_targets=sum(replay_ok.values()),
        b1_regular_auc=d.loc["B1", "AUC"],
        b2_regular_auc=d.loc["B2", "AUC"],
        b1_regular_ci_low=ci.loc["B1", "lo95"],
        b2_regular_ci_low=ci.loc["B2", "lo95"],
        b2_primary_q=b2.q_primary.iloc[0],
        mean_regular_pr_auc=d.PR_AUC.mean(),
        mean_regular_pr_gain=(d.PR_AUC - a.PR_AUC).mean(),
        all_acceptance=bool(
            all(replay_ok.values())
            and ci.lo95.gt(0).all()
            and b2.q_primary.iloc[0] < 0.05
        ),
    )


def freeze(panel, candidate):
    out = FINAL_RESULTS
    fields = HARD_GROUPS["D"] if candidate == "v4_hard" else GROUPS["D"]
    fitter = (
        fit_hard_route if candidate == "v4_hard" else partial(fit_route, candidate=candidate)
    )
    models = {
        target: fitter(with_target(derive(panel), target), fields, target)
        for target in ["B1", "B2"]
    }
    baselines = {
        target: fitter(with_target(panel, target), GROUPS["A"], target)
        for target in ["B1", "B2"]
    }
    previous = read_manifest(BENCHMARK_RESULTS)
    paths = [ROOT / name for name in previous["data_and_code"] if not name.startswith("src/")]
    paths.extend(
        BENCHMARK_RESULTS / f"conservative_{symbol}_features.parquet" for symbol in CORE
    )
    metadata = dict(
        phase="a",
        model_variant=candidate,
        cut_off=FREEZE,
        training_dates=[panel.date.min(), panel.date.max()],
        symbols=sorted(CORE),
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        feature_clock="conservative",
        selection="Prioritize all gates, replay checks, then conservative regular-day mean PR-AUC.",
    )
    publish_bundle(out, models, metadata, paths, baseline=baselines)


def run():
    FINAL_RESULTS.mkdir(parents=True, exist_ok=True)
    before = digest(CACHE / "usage.jsonl")
    panels = make_panels()
    granger = granger_family(panels["conservative"])
    granger.to_csv(FINAL_RESULTS / "pooled_granger.csv", index=False)
    raw = market_frame("NVDA", qqq=price_panel("halfday", "QQQ", continuous=True))
    feed = pd.read_csv(NEWS_INDEX, low_memory=False)
    news = timestamp_news(feed[feed.symbol.eq("NVDA")], "conservative")
    context = inputs("NVDA")
    items = {}
    summary = []
    original = pd.read_parquet(BENCHMARK_RESULTS / "predictions.parquet")
    original_case = pd.read_csv(BENCHMARK_RESULTS / "event_replay.csv")
    old_audit = json.loads((BENCHMARK_RESULTS / "audit.json").read_text(encoding="utf8"))
    items["v4_hard"] = (
        original,
        original_case,
        pd.DataFrame(),
        old_audit["max_tree_shap_reconstruction_error"],
    )
    for name in CANDIDATES:
        items[name] = collect_candidate(name, panels, raw, news, context)
    for name, (pred, replay, _, _) in items.items():
        folder = FINAL_RESULTS / "candidates" / name
        folder.mkdir(parents=True, exist_ok=True)
        report_tables(pred, folder)
        replay.to_csv(folder / "event_replay.csv", index=False)
        summary.append(
            criterion(
                name,
                pd.read_csv(folder / "earnings_split.csv"),
                pd.read_csv(folder / "paired_auc.csv"),
                replay,
                granger,
            )
        )
    comparison = pd.DataFrame(summary).sort_values(
        [
            "all_acceptance",
            "strict_targets",
            "mean_regular_pr_auc",
            "mean_regular_pr_gain",
        ],
        ascending=False,
    )
    comparison.to_csv(FINAL_RESULTS / "comparison.csv", index=False)
    best = comparison.candidate.iloc[0]
    pred, replay, _, error = items[best]
    assert (pred.last_train_end <= pred.cut).all()
    assert not pred.duplicated(
        ["clock", "scope", "target", "group", "date", "symbol"]
    ).any()
    pred.to_parquet(FINAL_RESULTS / "predictions.parquet", index=False)
    table = report_tables(pred, FINAL_RESULTS)
    replay.to_csv(FINAL_RESULTS / "event_replay.csv", index=False)
    pd.concat(
        [
            audit.assign(candidate=name)
            for name, (_, _, audit, _) in items.items()
            if not audit.empty
        ]
    ).to_csv(FINAL_RESULTS / "linear_weights.csv", index=False)
    dashboard(table, replay, FINAL_RESULTS, "pooled_3")
    freeze(panels["conservative"], best)
    after = digest(CACHE / "usage.jsonl")
    assert before == after
    audit = dict(
        selected=best,
        new_api_calls=0,
        usage_before=before,
        usage_after=after,
        cutoff_checks="passed",
        predictions=len(pred),
        max_tree_shap_reconstruction_error=error,
    )
    (FINAL_RESULTS / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    print(comparison.to_string(index=False), flush=True)
