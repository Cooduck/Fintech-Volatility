"""Run the historical benchmark and build comparable result tables."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from artifacts import digest, publish_bundle
from config import ARCHIVE_DATA as DATA, BENCHMARK_RESULTS, CACHE, CORE, DATES, FREEZE, NAMES, NEWS_INDEX, PORTFOLIO_ARCHIVE, ROOT
from evaluation import folds, metrics, paired, panel_granger, qualified_symbols
from features import build_features, inputs, market_frame, timestamp_news, with_target
from market_data import announcement, price_panel
from models import (
    HARD_GROUPS,
    walk_forward,
    route_scores,
    components,
    fit_hard_route,
)

def output_dir(phase):
    return BENCHMARK_RESULTS if phase == "a" else PORTFOLIO_ARCHIVE


def replay_case(route, raw, feed, context, derive=None):
    event = announcement()
    event_at = pd.Timestamp(event["utc"])
    opening = pd.Timestamp("2026-11-12T14:30Z")
    frame = raw[raw.cut.le(opening)].copy()
    assert frame.cut.iloc[-1] == opening and frame.session.iloc[-1] == "AM"
    assert len(feed[feed.url.eq(event["url"]) & feed["asof"].eq(event_at)]) == 1
    rows = []
    for cut in [
        event_at - pd.Timedelta(minutes=1),
        event_at,
        event_at + pd.Timedelta(minutes=1),
        opening,
    ]:
        frame.loc[frame.index[-1], "cut"] = cut
        current = build_features(frame, feed, *context).iloc[[-1]]
        if derive is not None:
            current = derive(current)
        point = current.iloc[0]
        parts = components(
            route["models"],
            current,
            route["fields"],
            route["target"],
            route.get("linear_fields"),
        )[0]
        risk = route_scores(current, route)[0]
        known = point.LatestTier1
        decay = (
            np.exp2(-(cut - known).total_seconds() / 3600) if pd.notna(known) else 0.0
        )
        # Classifier overlay preserves the V3 bounded scale. The regression
        # display adds 0.15 validation-score units and uses no probability lift.
        bonus = 0.15 * decay * ((1 - risk) if route["target"] == "B1" else 1.0)
        rows.append(
            dict(
                target=route["target"],
                cut=cut,
                linear=parts[0],
                tree=parts[1],
                score=risk,
                display_score=risk + bonus,
                latest_tier1=known,
                latest_news=point.LatestNews,
                cluster_impact=point.ClusterImpactSum,
                count=point.UniqueEventCount,
                event_brief=bool(point.HasLevel3Event),
                official_arrival=cut == event_at,
                official_url=event["url"] if cut >= event_at else "",
            )
        )
    return rows


def audit_quality(frame, feed, symbol, clock, halfdays):
    windows = halfdays[halfdays.date.isin(frame.date.unique())]
    times = feed["asof"]
    news_counts = times.searchsorted(windows.cut, side="right") - times.searchsorted(
        windows.start, side="right"
    )
    coverage = (news_counts > 0).mean()
    results = []
    for fold, days in enumerate(folds(DATES), 1):
        earlier = frame[frame.date.lt(min(days))]
        train = earlier.end.le(frame.loc[frame.date.isin(days), "cut"].min())
        earlier = earlier[train]
        seen = np.zeros(len(feed), bool)
        for start, cut in zip(earlier.start, earlier.cut):
            lo, hi = feed["asof"].searchsorted([start, cut], side="right")
            seen[lo:hi] = True
        articles = feed[seen]
        results.append(
            dict(
                clock=clock,
                symbol=symbol,
                fold=fold,
                training_news=len(articles),
                training_material=(
                    articles.event_type.ne("market_noise") & articles.impact_level.ge(2)
                ).mean(),
                coverage=coverage,
                coverage_am=frame.TotalNews.gt(0).mean(),
                negative_all=(feed.sentiment_score < -0.2).mean(),
            )
        )
    return results


def granger_family(panel):
    rows = []
    for target in ["B1", "B2"]:
        tests = panel_granger(
            with_target(panel, target), ["ClusterImpactSum", "SentStd"]
        )
        for test in tests:
            rows.append(
                dict(
                    test,
                    target=target,
                    family="primary_preregistered"
                    if test["lag"] == 2
                    else "exploratory",
                )
            )
    result = pd.DataFrame(rows)
    result["q_BH"] = result.groupby("family").p.transform(
        lambda p: multipletests(p, method="fdr_bh")[1]
    )
    result["q_primary"] = result.q_BH.where(result.family.eq("primary_preregistered"))
    assert result.family.eq("primary_preregistered").sum() == 4
    return result


def report_tables(predictions, out):
    grouped = ["clock", "scope", "target"]
    rows = []
    intervals = []
    slices = []
    for values, block in predictions.groupby(grouped):
        tags = dict(zip(grouped, values))
        days = sorted(block.date.unique())
        for segment, sub in [
            ("all", block),
            ("regular", block[block.is_earnings_window.eq(0)]),
            ("earnings", block[block.is_earnings_window.eq(1)]),
        ]:
            intervals.extend(dict(tags, slice=segment, **x) for x in paired(sub, days))
            for name in ["A", "C", "D"]:
                stats = dict(
                    tags, slice=segment, group=name, **metrics(sub[sub.group.eq(name)])
                )
                slices.append(stats)
                if segment == "all":
                    rows.append(stats)
    ablation = pd.DataFrame(rows)
    ablation.to_csv(out / "ablation.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "paired_auc.csv", index=False)
    pd.DataFrame(slices).to_csv(out / "earnings_split.csv", index=False)
    return ablation


def dashboard(table, case, out, scope):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    profile = table.query("clock=='conservative' and scope==@scope")
    for target, part in profile.groupby("target"):
        axes[0].plot(part.group, part.AUC, "o-", label=target)
    axes[0].set(
        xlabel="Feature group",
        ylabel="ROC-AUC",
        title="Pooled morning, conservative clock",
    )
    axes[0].legend()
    for ax, (target, part) in zip(axes[1:], case.groupby("target")):
        x = (
            pd.to_datetime(part.cut, utc=True)
            .dt.tz_convert("America/New_York")
            .dt.strftime("%H:%M")
        )
        ax.plot(x, part.score, "o-", label="Raw score")
        ax.plot(x, part.display_score, "o-", label="With event overlay")
        ax.axvline(1, color="gray", ls=":", label="Official release")
        ax.set(
            xlabel="Nov 12 ET snapshots", ylabel="Score", title=f"NVDA {target} replay"
        )
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(out / "dashboard.png", dpi=160)
    plt.close(fig)


def freeze(panel, out, phase, source_paths):
    dates = sorted(panel.date.unique())
    models = {
        target: fit_hard_route(with_target(panel, target), HARD_GROUPS["D"], target)
        for target in ("B1", "B2")
    }
    metadata = dict(
        phase=phase,
        cut_off=FREEZE,
        training_dates=[dates[0], dates[-1]],
        symbols=sorted(panel.symbol.unique()),
        created_utc=pd.Timestamp.now(tz="UTC").isoformat(),
        feature_clock="conservative",
    )
    publish_bundle(out, models, metadata, [ROOT / "requirements.txt", *source_paths])


def run(phase, index, *, out=None, freeze_models=True):
    out = output_dir(phase) if out is None else out
    out.mkdir(parents=True, exist_ok=True)
    symbols = CORE if phase == "a" else list(NAMES)
    rows = []
    quality = []
    replay = []
    shap_errors = []
    before = digest(CACHE / "usage.jsonl")
    contexts = {s: inputs(s) for s in symbols}
    qqq = price_panel("halfday", "QQQ", continuous=True)
    raw = {symbol: market_frame(symbol, qqq=qqq) for symbol in symbols}
    scope_main = "pooled_3" if phase == "a" else "full_9"
    for clock in ["conservative", "bounded"]:
        parts = []
        for symbol in symbols:
            feed = timestamp_news(index[index.symbol.eq(symbol)], clock)
            frame = build_features(raw[symbol], feed, *contexts[symbol])
            frame["y"] = (frame.z > 1.5).astype(int)
            frame = frame[frame.date.isin(DATES)].dropna(subset=HARD_GROUPS["D"] + ["z", "z_intraday"])
            if frame.empty:
                raise ValueError(f"No {symbol} rows on comparison dates")
            frame.to_parquet(out / f"{clock}_{symbol}_features.parquet", index=False)
            quality.extend(audit_quality(frame, feed, symbol, clock, raw[symbol]))
            parts.append(frame)
        panel = pd.concat(parts, ignore_index=True).sort_values(["cut", "symbol"])
        qualified = qualified_symbols(pd.DataFrame(quality), clock)
        scopes = {scope_main: panel}
        if qualified:
            scopes["qualified_only"] = panel[panel.symbol.isin(qualified)]
        scopes.update({symbol: panel[panel.symbol.eq(symbol)] for symbol in symbols})
        for scope, frame in scopes.items():
            for target in ["B1", "B2"]:
                callback = None
                if clock == "conservative" and scope == "NVDA":

                    def callback(route):
                        feed = timestamp_news(index[index.symbol.eq("NVDA")], clock)
                        replay.extend(
                            replay_case(route, raw["NVDA"], feed, contexts["NVDA"])
                        )

                pred, error = walk_forward(
                    with_target(frame, target), DATES, target, callback,
                    groups=HARD_GROUPS, route_fitter=fit_hard_route,
                )
                shap_errors.append(error)
                rows.append(pred.assign(clock=clock, scope=scope, target=target))
        if clock == "conservative":
            granger_family(panel).to_csv(out / "pooled_granger.csv", index=False)
        if clock == "conservative" and freeze_models:
            source_paths = [
                NEWS_INDEX,
                DATA / "VIX.csv",
                DATA / "earnings_notices.csv",
                ROOT / "data" / "lexicon.csv",
            ]
            for symbol in [*symbols, "QQQ"]:
                for interval in ["1h", "1d"]:
                    path = DATA / f"{symbol}_{interval}.csv"
                    source_paths.append(
                        path if path.exists() else ROOT / "data" / path.name
                    )
                if symbol != "QQQ":
                    source_paths.append(DATA / f"{symbol}_earnings.csv")
            if phase == "b":
                source_paths.append(out / "scored_news.csv.gz")
            freeze(panel, out, phase, source_paths)
        print(phase, clock, "finished", flush=True)
    pred = pd.concat(rows, ignore_index=True)
    assert not pred.duplicated(
        ["clock", "scope", "target", "group", "symbol", "date"]
    ).any()
    assert (pred.last_train_end <= pred.cut).all()
    pred.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(quality).to_csv(out / "corpus_quality.csv", index=False)
    ablation = report_tables(pred, out)
    case = pd.DataFrame(replay)
    case.to_csv(out / "event_replay.csv", index=False)
    dashboard(ablation, case, out, scope_main)
    after = digest(CACHE / "usage.jsonl")
    assert before == after
    audit = dict(
        phase=phase,
        symbols=symbols,
        predictions=len(pred),
        sha256_usage_before=before,
        sha256_usage_after=after,
        max_tree_shap_reconstruction_error=max(shap_errors),
        shap_scale="standalone tree raw score, separate from blended output",
        cutoff_checks="passed",
    )
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
