"""Chronological splits, ranking metrics and paired statistical inference."""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def folds(dates):
    """Use four expanding-window tests, then the reserved November interval."""
    dates = sorted(set(dates))
    ordinary = [date for date in dates if date < "2026-11-02"]
    return [list(part) for part in np.array_split(ordinary[len(ordinary) // 2:], 4)] + [
        [date for date in dates if date >= "2026-11-02"]
    ]


def qualified_symbols(quality, clock):
    sample = quality[quality.clock.eq(clock)]
    eligible = (
        sample.coverage.ge(0.70)
        & sample.negative_all.ge(0.15)
        & sample.training_material.ge(0.10)
    )
    flags = eligible.groupby(sample.symbol).all()
    return list(flags[flags].index)


def performance(table):
    ranked = table.sort_values(
        ["score", "cut", "symbol"], ascending=[False, True, True]
    )
    month = ranked.groupby(ranked.date.str[:7])
    rank, size = month.cumcount(), month.y.transform("size")
    top15 = ranked[rank < np.floor(0.15 * size)]
    top5 = ranked[rank < np.floor(0.05 * size)]
    hits = (table.y.astype(bool) & table.alert).sum()
    n_pos = table.y.sum()
    return dict(
        n=len(table),
        positive=n_pos,
        AUC=roc_auc_score(table.y, table.score),
        PR_AUC=average_precision_score(table.y, table.score),
        Recall=hits / n_pos,
        Precision=hits / max(table.alert.sum(), 1),
        alerts=table.alert.sum(),
        Recall_Top15=top15.y.sum() / n_pos,
        Lift_Top5=top5.y.mean() / table.y.mean(),
        Brier=brier_score_loss(table.y, table.score)
        if table.is_probability.all()
        else np.nan,
        statistical_briefs=table.stat_brief.sum(),
        event_briefs=table.event_brief.sum(),
        all_briefs=table.brief.sum(),
        brief_hits=(table.y.astype(bool) & table.brief).sum(),
    )


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


def metrics(table):
    if table.y.nunique() != 2:
        return dict(
            n=len(table), positive=int(table.y.sum()), AUC=np.nan, PR_AUC=np.nan
        )
    return performance(table)


def auc_contrasts(truth, values, sessions, contrast, repetitions=1000):
    """Estimate paired AUC contrasts by resampling five-session blocks."""
    truth = np.asarray(truth, dtype=bool)

    def statistic(positions):
        label = truth[positions]
        positive = label.sum()
        negative = len(label) - positive
        if not positive or not negative:
            return None
        ranks = rankdata(values[positions], axis=0)
        auc = (ranks[label].sum(axis=0) - positive * (positive + 1) / 2) / (positive * negative)
        return contrast(auc)

    point = statistic(np.arange(len(truth)))
    if point is None:
        return None, None, None, 0
    rng = np.random.default_rng(20261119)
    size = len(sessions)
    samples = []
    for _ in range(repetitions):
        starts = rng.integers(size, size=(size + 4) // 5)
        draw = ((starts[:, None] + np.arange(5)) % size).ravel()[:size]
        estimate = statistic(np.concatenate([sessions[i] for i in draw]))
        if estimate is not None:
            samples.append(estimate)
    if not samples:
        return point, np.full(len(point), np.nan), np.full(len(point), np.nan), 0
    lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
    return point, lower, upper, len(samples)


def paired(table, dates, repetitions=1000, *, baselines=("A", "C")):
    pivot = table.pivot(
        index=["date", "symbol", "y"], columns="group", values="score"
    ).reset_index()
    groups = pivot.groupby("date").indices
    sessions = [groups.get(date, np.empty(0, dtype=int)) for date in dates]
    # Keep empty subsamples reportable even when the expected columns are absent.
    values = pivot.reindex(columns=[*baselines, "D"]).to_numpy()
    point, lower, upper, count = auc_contrasts(
        pivot.y.to_numpy(), values, sessions,
        lambda auc: auc[-1] - auc[:-1],
        repetitions,
    )
    return [
        dict(
            contrast="D-" + baseline,
            delta_auc=point[i] if point is not None else np.nan,
            lo95=lower[i] if lower is not None else np.nan,
            hi95=upper[i] if upper is not None else np.nan,
            valid_draws=count,
        )
        for i, baseline in enumerate(baselines)
    ]
