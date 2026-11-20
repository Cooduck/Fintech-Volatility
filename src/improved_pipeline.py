"""Run the revision on frozen baseline dates. Outputs stay in results/improved."""

import argparse
from functools import lru_cache
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from collect_data import DATA, ROOT
from collect_finance import SYMBOLS
from panels import attach, price_panel, load_news
from semantic import score, key, ANCHORED, SYSTEM
from risk_evaluation import evaluate, performance, MARKET
from analysis_models import folds, paired_auc, granger

OUT = ROOT / "results" / "improved"
OUT.mkdir(exist_ok=True)
DATES = sorted(pd.read_csv(ROOT / "results" / "aligned_daily_decay.csv").date.unique())


def news_input(symbol, phase):
    if phase == "cached":
        return load_news("recent"), SYSTEM
    news = pd.read_csv(DATA / f"{symbol}_finance.csv")
    cnbc_file = DATA / "cnbc_finance.csv"
    if not cnbc_file.exists():
        raise FileNotFoundError("请先运行 cnbc_archive.py 获取 CNBC 快照")
    cnbc = pd.read_csv(cnbc_file)
    news = pd.concat(
        [
            news,
            cnbc.loc[
                cnbc.symbol == symbol, ["utc", "published", "title", "url", "source"]
            ],
        ],
        ignore_index=True,
    )
    news.utc = pd.to_datetime(news.utc, utc=True, format="mixed")
    news = (
        news.drop_duplicates(["utc", "title"]).sort_values("utc").reset_index(drop=True)
    )
    news.title = symbol + " | " + news.title
    return news, ANCHORED


@lru_cache(None)
def prices(grain, symbol, start="2024-11-21"):
    a = price_panel(grain, symbol, continuous=True)
    a = a[a.date >= start].copy()
    return a


def market_panel(grain, symbol, start="2024-11-21"):
    a = prices(grain, symbol, start).copy()
    q = prices(grain, "QQQ", start)
    q = q[
        ["cut", "session", "vol5", "vol20", "ret1", "sigma_lag1", "market_known_at"]
    ].rename(
        columns={
            v: "qqq_" + v
            for v in ["vol5", "vol20", "ret1", "sigma_lag1", "market_known_at"]
        }
    )
    a = a.merge(q, on=["cut", "session"], validate="one_to_one")
    a["market_known_at"] = a[["market_known_at", "qqq_market_known_at"]].max(axis=1)
    a["rel_vol_ratio"] = a.sigma_lag1 / a.qqq_sigma_lag1.replace(0, np.nan)
    a[MARKET] = a[MARKET].round(7).astype("float32")
    return a


def quality(panel, n, cache, sys, symbol, phase):
    s = pd.DataFrame([cache[key(t, sys)] for t in n.title])
    s["utc"] = n.utc.to_numpy()
    full = panel[panel.date.isin(DATES)]
    rows = []
    for fold, days in enumerate(folds(DATES, True), 1):
        test = full[full.date.isin(days)]
        tr = full[(full.end <= test.cut.min()) & (full.date < min(days))]
        # Count only texts available in these training windows, including weekend news.
        keep = np.zeros(len(n), bool)
        for start, end in zip(tr.start, tr.cut):
            lo, hi = n.utc.searchsorted([start, end], side="right")
            keep[lo:hi] = True
        used = s.loc[keep]
        coverage = tr.has_news.mean()
        neg = (used.sentiment_score < -0.2).mean()
        impact = ((used.impact_level >= 2) & used.event_type.ne("market_noise")).mean()
        rows.append(
            dict(
                symbol=symbol,
                phase=phase,
                fold=fold,
                training_windows=len(tr),
                training_articles=len(used),
                coverage=coverage,
                negative=neg,
                material=impact,
                pass_gate=coverage >= 0.70 and neg >= 0.15 and impact >= 0.10,
            )
        )
    counts = (
        s.groupby(["impact_level", "event_type"]).size().rename("count").reset_index()
    )
    counts.to_csv(OUT / f"{phase}_{symbol}_score_distribution.csv", index=False)
    return rows


def panel_granger(frames, features=("NewsShock", "SentStd", "ImpactMean")):
    import statsmodels.formula.api as smf
    from statsmodels.tsa.stattools import adfuller

    rows = []
    for session, b in frames.groupby("session"):
        for feature in features:
            z = (
                b[["symbol", "date", "sigma", feature]]
                .sort_values(["symbol", "date"])
                .copy()
            )
            for col in ["sigma", feature]:
                z[col] = z.groupby("symbol")[col].transform(
                    lambda x: (
                        x.diff()
                        if adfuller(x, autolag="BIC", result_object=False)[1] >= 0.05
                        else x
                    )
                )
            for lag in [1, 2, 3]:
                lagvars = []
                for k in range(1, lag + 1):
                    for col in ["sigma", feature]:
                        v = f"{col}_{k}"
                        z[v] = z.groupby("symbol")[col].shift(k)
                        lagvars.append(v)
                complete = z.dropna(subset=["sigma"] + lagvars)
                fit = smf.ols(
                    "sigma ~ C(symbol) + " + " + ".join(lagvars), data=complete
                ).fit(cov_type="cluster", cov_kwds={"groups": complete.date})
                f = fit.f_test(", ".join(f"{feature}_{k}=0" for k in range(1, lag + 1)))
                rows.append(
                    dict(
                        session=session,
                        feature=feature,
                        lag=lag,
                        F=float(f.fvalue),
                        p=float(f.pvalue),
                        n=len(complete),
                    )
                )
    return rows


def run(phase, offline, dry):
    symbols = ["NVDA"] if phase == "cached" else list(SYMBOLS)
    feeds = {s: news_input(s, phase) for s in symbols}
    system = feeds["NVDA"][1]
    texts = [t for n, _ in feeds.values() for t in n.title]
    est = score(texts, True, system)
    print(phase, est, flush=True)
    if dry:
        return
    if offline and est["uncached"]:
        raise ValueError("缺少评分缓存；真实模式会补评")
    cache = score(texts, system=system)
    if phase == "finance":
        ledgers = []
        for symbol, (n, sys) in feeds.items():
            rows = pd.DataFrame([cache[key(t, sys)] for t in n.title])
            fields = [
                "sentiment_score",
                "confidence",
                "impact_level",
                "event_type",
                "rationale",
                "key",
            ]
            ledgers.append(
                pd.concat([n.reset_index(drop=True), rows[fields]], axis=1).assign(
                    symbol=symbol
                )
            )
        pd.concat(ledgers, ignore_index=True).to_csv(
            OUT / "finance_news_index.csv.gz", index=False
        )
    prepared = {}
    health = []
    stats = []
    pooled_stats = []
    allmetrics = []
    allcase = []
    boot = []
    for symbol in symbols:
        n, _ = feeds[symbol]
        for grain in ["halfday", "daily"]:
            raw = market_panel(grain, symbol)
            x = attach(raw, n, cache, 0.25, system=system).dropna(subset=MARKET)
            x = x[x.date.isin(DATES)].assign(symbol=symbol)
            if grain == "halfday":
                health += quality(x, n, cache, system, symbol, phase)
            for target in ["B1", "B2"] if grain == "halfday" else ["daily"]:
                frame = x.copy()
                if target == "B2":
                    frame["z"] = x.z_intraday
                    frame["sigma"] = x.sigma_intraday
                    frame["y"] = (frame.z > 1.5).astype(int)
                prepared[symbol, target] = frame
                stats.extend(
                    dict(r, target=target) for r in granger(frame, symbol, grain)
                )
            x.to_csv(OUT / f"{phase}_{symbol}_{grain}.csv", index=False)
    pd.DataFrame(health).to_csv(OUT / f"{phase}_quality.csv", index=False)
    from revised_cases import capture, finish_cases

    for scope in ["NVDA"] if phase == "cached" else ["NVDA", "pooled"]:
        for target in ["B1", "B2", "daily"]:
            tickers = symbols if scope == "pooled" else ["NVDA"]
            frame = (
                pd.concat([prepared[s, target] for s in tickers])
                .sort_values(["cut", "symbol"])
                .reset_index(drop=True)
            )
            if scope == "pooled":
                pooled_stats.extend(
                    dict(r, target=target) for r in panel_granger(frame)
                )

            def callback(fold, model, cols, tr, te, p, spec):
                if scope == "NVDA":
                    allcase.extend(
                        capture(
                            phase,
                            target,
                            fold,
                            model,
                            cols,
                            tr,
                            te,
                            p,
                            spec,
                            feeds["NVDA"][0],
                            cache,
                            system,
                            market_panel,
                        )
                    )

            pred = evaluate(frame, DATES, callback)
            pred.to_parquet(
                OUT / f"{phase}_{scope}_{target}_predictions.parquet", index=False
            )
            tags = ["group", "algorithm", "balanced", "calibration"]
            for sl, subset in [("all", pred)] + (
                [("AM", pred[pred.session.eq("AM")])] if target != "daily" else []
            ):
                for k, d in subset.groupby(tags):
                    allmetrics.append(
                        dict(
                            scope=scope,
                            target=target,
                            slice=sl,
                            **dict(zip(tags, k)),
                            **performance(d),
                        )
                    )
            # Preserve all tickers inside each resampled date block.
            main = pred.query(
                "calibration=='raw' and not balanced and group!='E'"
            ).copy()
            main["session"] = main.symbol + "_" + main.session
            main["dataset"] = scope
            main["grain"] = target
            boot.append(paired_auc(main))
            print("evaluated", phase, scope, target, flush=True)
    for name, rows in [("granger", stats), ("pooled_granger", pooled_stats)]:
        if rows:
            g = pd.DataFrame(rows)
            g["q_BH"] = multipletests(g.p, method="fdr_bh")[1]
            g.to_csv(OUT / f"{phase}_{name}.csv", index=False)
    pd.concat(boot).to_csv(OUT / f"{phase}_paired_auc.csv", index=False)
    table = pd.DataFrame(allmetrics)
    gates = pd.DataFrame(health)
    table["training_corpus_qualified"] = table.scope.map(
        {
            "NVDA": bool(gates.loc[gates.symbol == "NVDA", "pass_gate"].all()),
            "pooled": bool(gates.pass_gate.all()),
        }
    )
    table.to_csv(OUT / f"{phase}_ablation.csv", index=False)
    finish_cases(phase, allcase, offline)


def historical_control():
    news = load_news("history")
    if score(news.title, True)["uncached"]:
        raise RuntimeError("历史对照需要原版冻结缓存")
    raw = market_panel("daily", "NVDA", "2019-01-02")
    raw = raw[raw.date <= "2023-12-29"]
    frame = (
        attach(raw, news, score(news.title), 0.25)
        .dropna(subset=MARKET)
        .assign(symbol="NVDA")
    )
    pred = evaluate(frame, sorted(frame.date.unique()), recent=False)
    pred.to_parquet(OUT / "historical_control_predictions.parquet", index=False)
    tags = ["group", "algorithm", "balanced", "calibration"]
    rows = [dict(**dict(zip(tags, k)), **performance(d)) for k, d in pred.groupby(tags)]
    pd.DataFrame(rows).to_csv(OUT / "historical_control_ablation.csv", index=False)
    g = pd.DataFrame(granger(frame, "NVDA", "daily"))
    g["q_BH"] = multipletests(g.p, method="fdr_bh")[1]
    g.to_csv(OUT / "historical_control_granger.csv", index=False)


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--phase", choices=["cached", "finance", "all"], default="all")
    a.add_argument("--mode", choices=["deepseek", "demo"], default="demo")
    a.add_argument("--dry-run", action="store_true")
    args = a.parse_args(argv)
    for phase in ["cached", "finance"] if args.phase == "all" else [args.phase]:
        run(phase, args.mode == "demo", args.dry_run)
        if phase == "cached" and not args.dry_run:
            historical_control()


if __name__ == "__main__":
    main()
