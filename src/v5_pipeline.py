"""Evaluate the V5 candidates and freeze one reproducible forward model."""

import argparse
from functools import partial
import json

import joblib
import pandas as pd

from collect_data import ROOT
from improved_pipeline import DATES
from panels import price_panel
from semantic import CACHE
from v3_features import timestamp_news
from v3_pipeline import INDEX, with_target
from v4_collect import CORE, FREEZE
from v4_features import inputs, market_frame
from v4_models import D as V4_FIELDS, fit_route as fit_v4, walk_forward
from v4_pipeline import (
    OUT as V4,
    dashboard,
    digest,
    granger_family,
    replay_case,
    report_tables,
)
from v5_models import GROUPS, derive, fit_route, weights

OUT = ROOT / "results" / "v5"
CANDIDATES = ("soft", "signed")


def qualify(clock):
    quality = pd.read_csv(V4 / "corpus_quality.csv").query("clock==@clock")
    qualifies = quality.assign(
        ok=quality.coverage.ge(0.70)
        & quality.negative_all.ge(0.15)
        & quality.training_material.ge(0.10)
    )
    flags = qualifies.groupby("symbol").ok.all()
    return list(flags[flags].index)


def make_panels():
    panels = {}
    for clock in ["conservative", "bounded"]:
        frames = [
            pd.read_parquet(V4 / f"{clock}_{symbol}_features.parquet")
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
    for clock, panel in panels.items():
        data = derive(panel)
        eligible = qualify(clock)
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
    out = OUT
    manifest_path = out / "frozen_manifest.json"
    history = ROOT / "results" / "v4_forward" / "predictions.jsonl"
    # Once real forward issuance begins, keep its existing model identity.
    if history.exists() and manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf8"))
        if any(
            json.loads(line).get("model_sha256") == old["model_sha256"]
            for line in history.read_text(encoding="utf8").splitlines()
            if line
        ):
            raise RuntimeError(
                "The current forward model already has issued predictions"
            )
    fields = V4_FIELDS if candidate == "v4_hard" else GROUPS["D"]
    fitter = (
        fit_v4 if candidate == "v4_hard" else partial(fit_route, candidate=candidate)
    )
    models = {
        target: fitter(with_target(derive(panel), target), fields, target)
        for target in ["B1", "B2"]
    }
    baselines = {
        target: fitter(with_target(panel, target), GROUPS["A"], target)
        for target in ["B1", "B2"]
    }
    path = out / "frozen_models.joblib"
    baseline_path = out / "frozen_baseline.joblib"
    joblib.dump(models, path)
    joblib.dump(baselines, baseline_path)
    previous = json.loads((V4 / "frozen_manifest.json").read_text(encoding="utf8"))
    files = set(previous["data_and_code"])
    files.update(
        ["src/v5_pipeline.py", "src/v5_models.py", "src/v4_forward_watcher.py"]
    )
    files.update(
        f"results/v4/conservative_{symbol}_features.parquet" for symbol in CORE
    )
    paths = {name: digest(ROOT / name) for name in sorted(files)}
    manifest = dict(
        phase="a",
        revision="v5",
        model_variant=candidate,
        cut_off=FREEZE,
        training_dates=[panel.date.min(), panel.date.max()],
        symbols=sorted(CORE),
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        feature_clock="conservative",
        model_sha256=digest(path),
        baseline_sha256=digest(baseline_path),
        data_and_code=paths,
        selection="Prioritize all gates, then the number of targets passing the mandatory raw-score replay, then conservative regular-day mean PR-AUC.",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf8")


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    before = digest(CACHE / "usage.jsonl")
    panels = make_panels()
    granger = granger_family(panels["conservative"])
    granger.to_csv(OUT / "pooled_granger.csv", index=False)
    raw = market_frame("NVDA", qqq=price_panel("halfday", "QQQ", continuous=True))
    feed = pd.read_csv(INDEX, low_memory=False)
    news = timestamp_news(feed[feed.symbol.eq("NVDA")], "conservative")
    context = inputs("NVDA")
    items = {}
    summary = []
    original = pd.read_parquet(V4 / "predictions.parquet")
    original_case = pd.read_csv(V4 / "event_replay.csv")
    old_audit = json.loads((V4 / "audit.json").read_text(encoding="utf8"))
    items["v4_hard"] = (
        original,
        original_case,
        pd.DataFrame(),
        old_audit["max_tree_shap_reconstruction_error"],
    )
    for name in CANDIDATES:
        items[name] = collect_candidate(name, panels, raw, news, context)
    for name, (pred, replay, _, _) in items.items():
        folder = OUT / "candidates" / name
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
    comparison.to_csv(OUT / "comparison.csv", index=False)
    best = comparison.candidate.iloc[0]
    pred, replay, _, error = items[best]
    assert (pred.last_train_end <= pred.cut).all()
    assert not pred.duplicated(
        ["clock", "scope", "target", "group", "date", "symbol"]
    ).any()
    pred.to_parquet(OUT / "predictions.parquet", index=False)
    table = report_tables(pred, OUT)
    replay.to_csv(OUT / "event_replay.csv", index=False)
    pd.concat(
        [
            audit.assign(candidate=name)
            for name, (_, _, audit, _) in items.items()
            if not audit.empty
        ]
    ).to_csv(OUT / "linear_weights.csv", index=False)
    dashboard(table, replay, OUT, "pooled_3")
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
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    print(comparison.to_string(index=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["a", "b"], default="a")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--offline", action="store_true", help="reproduce the completed Phase B backtest locally")
    parser.add_argument(
        "--approved-usd",
        type=float,
        help="total additional Phase B budget, including retries and resumed runs",
    )
    args = parser.parse_args(argv)
    if args.offline and (args.phase != "b" or args.approved_usd is not None):
        parser.error("--offline requires --phase b and cannot be combined with --approved-usd")
    if args.phase == "b":
        from v5_phase_b import main as phase_b

        return phase_b(args)
    if args.dry_run:
        print("Three-stock V5 comparison uses only the frozen V4 input. No API calls.")
        return
    run()


if __name__ == "__main__":
    main()
