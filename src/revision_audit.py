"""Validate revision outputs and compare performance on identical targets and dates."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from panels import TECH, SEM
from analysis_models import classifiers, OUT as BASE
from improved_pipeline import OUT, news_input
from risk_evaluation import policy, performance
from collect_data import ROOT
from semantic import read, key


def old_policy():
    df = pd.read_csv(BASE / "aligned_session_am_pm.csv")
    pred = (
        pd.read_csv(BASE / "oos_predictions.csv")
        .query(
            "dataset=='recent' and grain=='halfday' and algorithm=='LightGBM' and group=='D'"
        )
        .copy()
    )
    for fold, part in pred.groupby("fold"):
        tr = df[(df.date < part.date.min()) & (df.end <= part.cut.min())]
        days = sorted(tr.date.unique())
        start = days[int(0.65 * len(days))]
        val = tr[tr.date >= start]
        inner = tr[(tr.date < start) & (tr.end <= val.cut.min())]
        cols = TECH + ["is_afternoon_session"] + SEM
        model = classifiers()["LightGBM"].fit(inner[cols], inner.y)
        spec = policy(model.predict_proba(val[cols])[:, 1], val.y, tr.y.mean())
        for k, v in spec.items():
            pred.loc[part.index, k] = v
    pred["symbol"] = "NVDA"
    pred["alert"] = pred.p >= pred.alert_line
    pred["brief"] = (pred.p >= pred.brief_line) | (pred.p / pred.prior >= 3.5)
    pred.to_csv(OUT / "original_scores_new_policy.csv", index=False)
    return pred


def main():
    old = old_policy()
    rows = []
    fixed = old.copy()
    fixed["alert"] = fixed.p >= 0.5
    fixed["brief"] = fixed.p > 0.7
    for label, g in [("original", fixed), ("threshold_only", old)]:
        rows.append(dict(stage=label, **performance(g)))
    for phase in ["cached", "finance"]:
        d = pd.read_parquet(OUT / f"{phase}_NVDA_B1_predictions.parquet").query(
            "algorithm=='LightGBM' and group=='D' and not balanced and calibration=='raw'"
        )
        paired = d.merge(
            old, on=["date", "session"], suffixes=("", "_old"), validate="one_to_one"
        )
        assert len(paired) == len(old) == len(d) and (paired.y == paired.y_old).all()
        rows.append(dict(stage=phase, **performance(d)))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "before_after.csv", index=False)
    from sklearn.metrics import roc_auc_score, average_precision_score

    pairs = paired.sort_values(["date", "session"]).reset_index(drop=True)
    clusters = list(pairs.groupby("date").indices.values())
    rng = np.random.default_rng(292)
    truth = pairs.y.to_numpy()
    new = pairs.p.to_numpy()
    base = pairs.p_old.to_numpy()
    contrasts = []
    for metric in [roc_auc_score, average_precision_score]:
        samples = []
        for _ in range(1000):
            starts = rng.integers(len(clusters), size=(len(clusters) + 4) // 5)
            block = ((starts[:, None] + np.arange(5)) % len(clusters)).ravel()[
                : len(clusters)
            ]
            idx = np.concatenate([clusters[i] for i in block])
            if len(set(truth[idx])) == 2:
                samples.append(
                    metric(truth[idx], new[idx]) - metric(truth[idx], base[idx])
                )
        lo, hi = np.percentile(samples, [2.5, 97.5])
        contrasts.append(
            dict(
                metric=metric.__name__,
                delta=metric(truth, new) - metric(truth, base),
                lo95=lo,
                hi95=hi,
            )
        )
    pd.DataFrame(contrasts).to_csv(OUT / "revised_vs_original.csv", index=False)
    checks = []
    for phase in ["cached", "finance"]:
        for file in OUT.glob(f"{phase}_*_predictions.parquet"):
            d = pd.read_parquet(file)
            assert not d.duplicated(
                [
                    "date",
                    "session",
                    "symbol",
                    "algorithm",
                    "group",
                    "balanced",
                    "calibration",
                ]
            ).any()
            assert (d.last_train_end <= d.cut).all() and (
                d.inner_train_end <= d.inner_val_first_cut
            ).all()
            assert np.array_equal(
                d.brief, (d.p >= d.brief_line) | (d.p / d.prior >= 3.5)
            )
            checks.append(
                dict(
                    file=file.name,
                    n=len(d),
                    folds=d.fold.nunique(),
                    first=d.date.min(),
                    last=d.date.max(),
                )
            )
        scores = pd.read_csv(OUT / f"{phase}_case_scores.csv")
        brief = pd.read_csv(OUT / f"{phase}_briefs.csv")
        assert (
            scores.shap_error.max() < 1e-5 and len(brief) == scores.brief_trigger.sum()
        )
        for sym in ["NVDA"] if phase == "cached" else ["NVDA", "MSFT", "TSLA"]:
            n, sys = news_input(sym, phase)
            for grain in ["halfday", "daily"]:
                x = pd.read_csv(OUT / f"{phase}_{sym}_{grain}.csv")
                cut = pd.to_datetime(x.cut, utc=True)
                left = pd.to_datetime(x.start, utc=True)
                assert np.array_equal(
                    x.N,
                    n.utc.searchsorted(cut, side="right")
                    - n.utc.searchsorted(left, side="right"),
                )
                if grain == "halfday":
                    assert (x.sigma_intraday <= x.sigma + 1e-10).all()
                    pm = x.session == "PM"
                    assert np.allclose(x.loc[pm, "sigma_intraday"], x.loc[pm, "sigma"])
                    assert (
                        left[x.session == "AM"]
                        .dt.tz_convert("America/New_York")
                        .dt.strftime("%H:%M")
                        == "12:30"
                    ).all()
    pd.DataFrame(checks).to_csv(OUT / "verification.csv", index=False)
    cache = {r["key"]: r for r in read("deepseek_cache.jsonl")}
    health_rows = []
    for sym in ["NVDA", "MSFT", "TSLA"]:
        n, system = news_input(sym, "finance")
        scores = pd.DataFrame([cache[key(t, system)] for t in n.title])
        x = pd.read_csv(OUT / f"finance_{sym}_halfday.csv")
        folds_health = pd.read_csv(OUT / "finance_quality.csv").query("symbol==@sym")
        coverage = x.has_news.mean()
        negative = (scores.sentiment_score < -0.2).mean()
        material = folds_health.material.min()
        health_rows.append(
            dict(
                symbol=sym,
                articles=len(n),
                windows=len(x),
                coverage=coverage,
                negative=negative,
                min_training_material=material,
                all_training_folds_pass=bool(folds_health.pass_gate.all()),
                corpus_pass=bool(
                    coverage >= 0.7 and negative >= 0.15 and material >= 0.1
                ),
            )
        )
    pd.DataFrame(health_rows).to_csv(OUT / "corpus_acceptance.csv", index=False)
    perf = []
    for target in ["B1", "B2", "daily"]:
        pool = pd.read_parquet(OUT / f"finance_pooled_{target}_predictions.parquet")
        for (sym, group, algo), d in pool.query(
            "not balanced and calibration=='raw'"
        ).groupby(["symbol", "group", "algorithm"]):
            perf.append(
                dict(
                    symbol=sym,
                    target=target,
                    group=group,
                    algorithm=algo,
                    **performance(d),
                )
            )
    pd.DataFrame(perf).to_csv(OUT / "pooled_by_symbol.csv", index=False)
    import hashlib, json

    original = json.loads((BASE / "audit.json").read_text(encoding="utf8"))
    usage = read("usage.jsonl")
    total = sum(row["estimated_usd"] for row in usage)
    news = pd.read_csv(OUT / "finance_news_index.csv.gz")
    published = pd.to_datetime(news.published, utc=True, format="mixed")
    available = pd.to_datetime(news.utc, utc=True, format="mixed")
    assert (available[published.notna()] >= published.dropna()).all()
    assert not news.duplicated(["symbol", "utc", "title"]).any()
    files = list((ROOT / "data").glob("*_finance.csv"))
    manifest = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    audit = dict(
        new_requests=len(usage) - original["calls"],
        new_estimated_usd=total - original["estimated_usd"],
        cumulative_estimated_usd=total,
        scored_news_rows=len(news),
        unique_scoring_keys=news.key.nunique(),
        source_sha256=manifest,
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
    health = pd.read_csv(OUT / "finance_quality.csv")
    prim = pd.read_csv(OUT / "finance_ablation.csv").query(
        "target=='B1' and slice=='all' and group in ['D','E'] and not balanced and calibration=='raw'"
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ax = axes[0, 0]
    for metric in ["Recall", "Precision", "alert_rate"]:
        ax.plot(
            range(4),
            summary[metric],
            marker="o",
            label={
                "Recall": "召回率",
                "Precision": "警报命中率",
                "alert_rate": "警报率",
            }[metric],
        )
    ax.set_xticks(range(4), ["原实验", "只改阈值", "再改模型与窗口", "换财经新闻"])
    ax.set(title="NVDA · 同一组 482 个样本外半日", ylim=(0, 1))
    ax.legend()
    ax = axes[0, 1]
    for sym, g in health.groupby("symbol"):
        ax.plot(g.fold, g.coverage, marker="o", label=sym)
    ax.axhline(0.70, ls="--", color="grey")
    ax.set(title="各训练折有新闻的半日占比", xlabel="训练折", ylim=(0, 1.03))
    ax.legend()
    ax = axes[1, 0]
    pivot = prim.pivot(index="algorithm", columns="scope", values="PR_AUC")
    pivot.plot.bar(ax=ax, rot=0)
    ax.set(title="各模型样本外平均精确率（PR-AUC）", ylabel="PR-AUC", xlabel="")
    ax = axes[1, 1]
    case = pd.read_csv(OUT / "finance_case_scores.csv").query(
        "date=='2026-11-12' and session=='AM'"
    )
    for target, g in case.groupby("target"):
        g = g.sort_values("cut")
        hours = pd.to_datetime(g.cut, utc=True).dt.tz_convert("America/New_York")
        ax.plot(hours, g.p, marker="o", label=target)
        ax.axhline(
            min(g.brief_line.iloc[0], 3.5 * g.prior.iloc[0]),
            ls=":",
            color=ax.lines[-1].get_color(),
        )
    from matplotlib.dates import DateFormatter

    ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz="America/New_York"))
    ax.axvline(pd.Timestamp("2026-11-12T13:48Z"), ls="--", color="gray")
    ax.set(
        title="11 月 12 日盘前回放 · 点线为各目标简报门槛",
        xlabel="美东时间（虚线为 08:48 公告）",
        ylabel="模型分数",
    )
    ax.legend()
    fig.savefig(OUT / "dashboard.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
