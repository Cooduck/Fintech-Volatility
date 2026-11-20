"""Offline V3 experiments using the frozen V2 finance index and price data."""

import argparse
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from collect_data import DATA, ROOT
from improved_pipeline import DATES, OUT as PREVIOUS, market_panel, panel_granger
from panels import TECH
from risk_evaluation import MARKET
from v3_features import SENTIMENT, TAIL, aggregate, timestamp_news
from v3_models import performance, scores, walk_forward

OUT = ROOT / "results" / "v3"
INDEX = PREVIOUS / "finance_news_index.csv.gz"
SYMBOLS = ["NVDA", "MSFT", "TSLA"]
SCORE_LOG = ROOT / "cache" / "usage.jsonl"
EVENT_DAY = "2026-11-12"


def align(raw, news, lexicon, profile):
    frame = aggregate(raw, news, lexicon, profile)
    frame["y"] = (frame.z > 1.5).astype(int)
    return frame.dropna(subset=TECH + MARKET + SENTIMENT + TAIL + ["z", "z_intraday"])


def timestamp_audit(conservative):
    delay = conservative.delay_hours
    result = dict(
        rows=len(conservative),
        market_noise=int(conservative.event_type.eq("market_noise").sum()),
        substantive=int(
            (
                conservative.event_type.ne("market_noise")
                & conservative.impact_level.ge(1)
            ).sum()
        ),
        any_delay=int(delay.gt(0).sum()),
        more_than_hour=int(delay.gt(1).sum()),
        delayed_hours_quantiles={
            str(q): float(delay[delay.gt(0)].quantile(q))
            for q in [0.5, 0.75, 0.9, 0.99]
        },
        actions=conservative.action.value_counts().to_dict(),
    )
    return result


def old_panel(symbol):
    frame = pd.read_csv(PREVIOUS / f"finance_{symbol}_halfday.csv")
    for col in ["start", "cut", "end", "market_known_at"]:
        frame[col] = pd.to_datetime(frame[col], utc=True).dt.as_unit("ns")
    frame[TECH + MARKET] = frame[TECH + MARKET].astype("float32")
    return frame


def with_target(frame, target):
    if target == "B1":
        return frame
    return frame.assign(
        z=frame.z_intraday,
        sigma=frame.sigma_intraday,
        y=(frame.z_intraday > 1.5).astype(int),
    )


def case_replay(
    fitted, fields, alert_line, brief_line, prior, raw, news, lexicon, target
):
    from collect_data import events

    official = events()[-1]
    announcement = pd.Timestamp(official["utc"])
    opening = pd.Timestamp(EVENT_DAY + "T14:30Z")
    source = news[(news.url == official["url"]) & news["asof"].eq(announcement)]
    assert (
        len(source) == 1
        and source.impact_level.iloc[0] == 3
        and source.source_weight.iloc[0] == 1
    )
    upto_open = raw[raw.cut <= opening].copy()
    assert upto_open.iloc[-1].cut == opening and upto_open.iloc[-1].session == "AM"
    decisions = [
        announcement - pd.Timedelta(minutes=1),
        announcement,
        announcement + pd.Timedelta(minutes=1),
        opening,
    ]
    rows = []
    for cutoff in decisions:
        upto_open.loc[upto_open.index[-1], "cut"] = cutoff
        current = align(upto_open, news, lexicon, "weighted").iloc[[-1]]
        row = current.iloc[0]
        assert row.market_known_at <= cutoff and row.LatestNews <= cutoff
        l2 = scores("L2", fitted["L2"], current, fields)[0]
        tree = scores("MonotoneLightGBM", fitted["MonotoneLightGBM"], current, fields)[
            0
        ]
        model_score = 0.7 * l2 + 0.3 * tree
        direct = pd.notna(row.LatestTier1)
        age = (cutoff - row.LatestTier1).total_seconds() / 3600 if direct else np.inf
        overlay = 0.15 * np.exp2(-age) * (1 - model_score)
        stat = model_score >= brief_line or model_score / prior >= 3.5
        official_now = announcement <= cutoff < announcement + pd.Timedelta(minutes=2)
        rows.append(
            dict(
                target=target,
                cutoff=cutoff,
                market_known_at=row.market_known_at,
                latest_news=row.LatestNews,
                latest_tier1=row.LatestTier1,
                n=row.N,
                total_news=row.TotalNews,
                has_level3=row.HasLevel3Event,
                impact_sum=row.SubstantiveImpactSum,
                min_sentiment=row.MinSentiment,
                l2=l2,
                tree=tree,
                model_score=model_score,
                alert_line=alert_line,
                brief_line=brief_line,
                stat_brief=stat,
                event_brief=direct,
                official_event_in_window=cutoff >= announcement,
                official_event_now=official_now,
                event_overlay=overlay,
                display_score=model_score + overlay,
                official_url=official["url"] if cutoff >= announcement else "",
                cached_rationale=source.rationale.iloc[0]
                if cutoff >= announcement
                else "",
            )
        )
    return rows


def event_notifications(frame, news):
    """Group simultaneous Tier-1 arrivals into one brief per ticker and timestamp."""
    rows = []
    eligible = news[
        (news.impact_level == 3)
        & (news.source_weight == 1)
        & news.event_type.ne("market_noise")
    ]
    for row in frame.itertuples(index=False):
        batch = eligible[eligible["asof"].gt(row.start) & eligible["asof"].le(row.cut)]
        for asof, articles in batch.groupby("asof"):
            rows.append(
                dict(
                    symbol=row.symbol,
                    date=row.date,
                    session=row.session,
                    known_at=asof,
                    session_cut=row.cut,
                    sources=" | ".join(articles.source.unique()),
                    titles=" | ".join(articles.title),
                    urls=" | ".join(articles.url.unique()),
                    brief="三级事件到达。"
                    + "；".join(articles.rationale.dropna().unique()),
                )
            )
    return rows


def scheduled_briefs(predictions):
    """Write local briefs for the main deployment policy at scheduled cutoffs."""
    selected = predictions.query(
        "clock=='conservative' and profile=='weighted' and algorithm=='Ensemble70' "
        "and target=='B1' and group=='D' and brief"
    )
    rows = []
    for row in selected.itertuples(index=False):
        reasons = []
        if row.stat_brief:
            if row.score >= row.brief_line:
                reasons.append(
                    f"集成分数 {row.score:.3f} 达到本折简报线 {row.brief_line:.3f}"
                )
            if row.score / row.prior >= 3.5:
                reasons.append(
                    f"集成分数为训练期异常率的 {row.score / row.prior:.2f} 倍"
                )
        if row.event_brief:
            reasons.append("窗口中已收到一级信源的三级事件")
        rows.append(
            dict(
                scope=row.scope,
                symbol=row.symbol,
                date=row.date,
                session=row.session,
                cutoff=row.cut,
                model_score=row.score,
                brief_line=row.brief_line,
                statistical=row.stat_brief,
                event=row.event_brief,
                y=row.y,
                brief="。".join(reasons) + "。",
            )
        )
    return pd.DataFrame(rows)


def prior_predictions(target):
    frames = []
    for scope in ["NVDA", "pooled"]:
        data = pd.read_parquet(
            PREVIOUS / f"finance_{scope}_{target}_predictions.parquet"
        )
        base = data.query(
            "calibration=='raw' and not balanced and algorithm=='L2'"
        ).copy()
        base = base.rename(columns={"p": "score"}).assign(
            clock="conservative",
            profile="v2_mixed",
            scope=scope,
            target=target,
            is_probability=True,
            stat_brief=base.brief,
            event_brief=False,
        )
        frames.append(base)
    return pd.concat(frames, ignore_index=True)


def charts(summary, case):
    # Portable English chart labels. The Markdown report explains the policies in Chinese.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    grid = summary.query(
        "scope=='pooled' and target=='B1' and clock=='conservative' and profile=='weighted'"
    )
    for session, block in grid[grid.algorithm.eq("Ensemble70")].groupby("session"):
        axes[0].plot(block.group, block.AUC, "o-", label=session)
    axes[0].set(
        xlabel="Feature group", ylabel="ROC-AUC", title="B1 pooled results by session"
    )
    axes[0].legend()
    for ax, target, note in zip(
        axes[1:], ["B1", "B2"], ["with opening gap", "post-open only"]
    ):
        segment = case[case.target.eq(target)]
        times = segment.cutoff.dt.tz_convert("America/New_York").dt.strftime("%H:%M")
        ax.plot(times, segment.model_score, "o-", label="Ensemble score")
        ax.plot(times, segment.display_score, "o-", label="With event overlay")
        ax.axvline(1, ls=":", color="gray", label="08:48 announcement")
        ax.set(
            xlabel="Nov 12 decision snapshots (ET)",
            ylabel="Score",
            title=f"NVDA {target} · {note}",
        )
        ax.legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(OUT / "dashboard.png", dpi=160)
    plt.close(fig)


def run():
    from v3_models import paired_intervals

    OUT.mkdir(parents=True, exist_ok=True)
    original_hash = (
        hashlib.sha256(SCORE_LOG.read_bytes()).hexdigest()
        if SCORE_LOG.exists()
        else None
    )
    index = pd.read_csv(INDEX)
    assert not index.duplicated(["symbol", "utc", "title"]).any()
    old = {symbol: old_panel(symbol) for symbol in SYMBOLS}
    lexicon = pd.read_csv(DATA / "lexicon.csv")
    raw = {
        symbol: market_panel("halfday", symbol).assign(symbol=symbol)
        for symbol in SYMBOLS
    }
    feeds = {
        clock: timestamp_news(index, clock) for clock in ["conservative", "bounded"]
    }
    parts = []
    tests = []
    replay = []
    all_events = []
    for clock, feed in feeds.items():
        for profile in (
            ["weighted", "filtered", "pm_material", "v2_split"]
            if clock == "conservative"
            else ["weighted"]
        ):
            frames = {}
            for symbol in SYMBOLS:
                articles = feed[feed.symbol.eq(symbol)].reset_index(drop=True)
                if profile == "v2_split":
                    frame = old[symbol]
                else:
                    frame = align(raw[symbol], articles, lexicon, profile)
                    frame = frame[frame.date.isin(DATES)].copy()
                    assert (
                        frame.LatestNews.isna() | frame.LatestNews.le(frame.cut)
                    ).all()
                    assert (frame.market_known_at <= frame.cut).all()
                    pd.testing.assert_frame_equal(
                        frame[["cut", "session", "y", "z"]].reset_index(drop=True),
                        old[symbol][["cut", "session", "y", "z"]].reset_index(
                            drop=True
                        ),
                        check_exact=False,
                        rtol=1e-9,
                        atol=1e-10,
                    )
                    if profile == "weighted":
                        frame.to_parquet(
                            OUT / f"{clock}_{symbol}_features.parquet", index=False
                        )
                        all_events.extend(
                            dict(evt, clock=clock)
                            for evt in event_notifications(frame, articles)
                        )
                frames[symbol] = frame
            for target in ["B1", "B2"] if profile == "weighted" else ["B1"]:
                combined = with_target(
                    pd.concat(frames.values(), ignore_index=True), target
                )
                if profile == "weighted":
                    subset = combined[combined.session.eq("AM")]
                    tests.extend(
                        dict(row, clock=clock, target=target)
                        for row in panel_granger(
                            subset, ["NewsShock", "SentStd", "SubstantiveImpactSum"]
                        )
                    )
                for scope in ["NVDA", "pooled"]:
                    sample = (
                        combined.query("symbol=='NVDA'")
                        if scope == "NVDA"
                        else combined
                    )
                    sessions = (
                        ["PM"]
                        if profile == "pm_material"
                        else ["AM"]
                        if target == "B2"
                        else ["AM", "PM"]
                    )
                    for session in sessions:
                        call = None
                        if (
                            profile == "weighted"
                            and scope == "NVDA"
                            and session == "AM"
                            and clock == "conservative"
                        ):
                            articles = feed[feed.symbol.eq("NVDA")].reset_index(
                                drop=True
                            )

                            def call(fits, fields, alert, brief, prior):
                                replay.extend(
                                    case_replay(
                                        fits,
                                        fields,
                                        alert,
                                        brief,
                                        prior,
                                        raw["NVDA"],
                                        articles,
                                        lexicon,
                                        target,
                                    )
                                )

                        model = walk_forward(
                            sample[sample.session.eq(session)], DATES, profile, call
                        )
                        parts.append(
                            model.assign(
                                clock=clock, profile=profile, scope=scope, target=target
                            )
                        )
            print(clock, profile, "complete", flush=True)
    for target in ["B1", "B2"]:
        base = prior_predictions(target)
        if target == "B2":
            base = base[base.session.eq("AM")]
        parts.append(base)
    predictions = pd.concat(parts, ignore_index=True)
    keys = ["clock", "profile", "scope", "target", "session", "algorithm", "group"]
    assert not predictions.duplicated(keys + ["date", "symbol"]).any()
    assert (predictions.last_train_end <= predictions.cut).all()
    assert (predictions.inner_train_end <= predictions.inner_val_first_cut).all()
    assert (
        predictions.brief == (predictions.stat_brief | predictions.event_brief)
    ).all()
    predictions.to_parquet(OUT / "predictions.parquet", index=False)
    scheduled_briefs(predictions).to_csv(OUT / "scheduled_briefs.csv", index=False)
    summary = pd.DataFrame(
        [dict(zip(keys, k), **performance(v)) for k, v in predictions.groupby(keys)]
    )
    summary.to_csv(OUT / "ablation.csv", index=False)
    paired_intervals(predictions).to_csv(OUT / "paired_auc.csv", index=False)
    granger = pd.DataFrame(tests)
    granger["q_BH"] = multipletests(granger.p, method="fdr_bh")[1]
    granger.to_csv(OUT / "pooled_granger.csv", index=False)
    case = pd.DataFrame(replay)
    case.to_csv(OUT / "event_replay.csv", index=False)
    pd.DataFrame(all_events).to_csv(OUT / "event_briefs.csv", index=False)
    charts(summary, case)
    final_hash = (
        hashlib.sha256(SCORE_LOG.read_bytes()).hexdigest()
        if SCORE_LOG.exists()
        else None
    )
    assert original_hash == final_hash
    audit = dict(
        **timestamp_audit(feeds["conservative"]),
        predictions=len(predictions),
        briefs=len(all_events),
        cutoff_checks="passed",
        input_sha256=hashlib.sha256(INDEX.read_bytes()).hexdigest(),
        usage_sha256=final_hash,
        new_api_calls=0,
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline V3 volatility experiment")
    parser.add_argument("--mode", choices=["demo"], default="demo")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args(argv)
    if opts.dry_run:
        print("V3 inputs ready:", INDEX.is_file(), "new API calls: 0")
        return
    run()


if __name__ == "__main__":
    main()
