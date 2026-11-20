"""AM and PM model comparisons with forward-only inner thresholds."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis_models import folds
from panels import TECH, SEM
from risk_evaluation import MARKET, estimator
from v3_features import SENTIMENT, TAIL, MONOTONE

CLASSIFIERS = ["L2", "ElasticNet", "MonotoneLightGBM", "Ensemble70"]
REGRESSORS = {"Ridge", "HuberLightGBM"}
TREE_SETTINGS = dict(
    n_estimators=80,
    learning_rate=0.035,
    max_depth=4,
    num_leaves=9,
    min_child_samples=12,
    min_data_in_bin=1,
    max_bin=127,
    reg_alpha=0.75,
    reg_lambda=5.0,
    random_state=24,
    n_jobs=2,
    verbosity=-1,
)


def columns(profile):
    base = TECH + MARKET
    if profile == "v2_split":
        sentiment = base + SEM[:6] + ["has_news"]
        full = base + SEM + ["has_news", "max_impact_level"]
    else:
        sentiment = base + SENTIMENT
        full = sentiment + TAIL
    return {"A": base, "B": base + ["Lexicon"], "C": sentiment, "D": full}


def create_model(name, fields):
    if name == "L2":
        return estimator("L2", False)
    if name == "ElasticNet":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5, l1_ratio=0.35, solver="saga", max_iter=4000, random_state=24
            ),
        )
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    if name == "HuberLightGBM":
        return LGBMRegressor(objective="huber", alpha=0.9, **TREE_SETTINGS)
    return LGBMClassifier(
        monotone_constraints=[int(col in MONOTONE) for col in fields], **TREE_SETTINGS
    )


def fit_model(name, fields, frame):
    model = create_model(name, fields)
    target = frame.z.clip(-2.0, 6.0) if name in REGRESSORS else frame.y
    return model.fit(frame[fields], target)


def scores(name, model, frame, fields):
    if name in REGRESSORS:
        return model.predict(frame[fields])
    return model.predict_proba(frame[fields])[:, 1]


def walk_forward(frame, dates, profile, callback=None):
    """Use identical outer test dates and a chronological inner validation segment."""
    results = []
    groups = columns(profile)
    names = ["L2"] if profile == "v2_split" else CLASSIFIERS + sorted(REGRESSORS)
    for fold, days in enumerate(folds(dates, True), 1):
        test = frame[frame.date.isin(days)]
        train = frame[(frame.date < min(days)) & (frame.end <= test.cut.min())]
        train_days = sorted(train.date.unique())
        split = train_days[int(0.65 * len(train_days))]
        validation = train[train.date >= split]
        early = train[(train.date < split) & (train.end <= validation.cut.min())]
        assert (
            early.end.max() <= validation.cut.min()
            and train.end.max() <= test.cut.min()
        )
        for group, fields in groups.items():
            predicted = {}
            fitted = {}
            for name in names:
                if name == "Ensemble70":
                    l2_val, l2_out = predicted["L2"]
                    tree_val, tree_out = predicted["MonotoneLightGBM"]
                    valid, out = (
                        0.7 * l2_val + 0.3 * tree_val,
                        0.7 * l2_out + 0.3 * tree_out,
                    )
                else:
                    inner = fit_model(name, fields, early)
                    model = fit_model(name, fields, train)
                    fitted[name] = model
                    valid = scores(name, inner, validation, fields)
                    out = scores(name, model, test, fields)
                predicted[name] = valid, out
                alert_line, brief_line = np.quantile(valid, [0.85, 0.95])
                prior = train.y.mean()
                probability = name not in REGRESSORS
                statistical = (out >= brief_line) | (
                    (out / prior >= 3.5) if probability else False
                )
                # Only D uses the explicit event feature and the independent event route.
                direct = (
                    test.HasLevel3Event.astype(bool).to_numpy()
                    if group == "D" and profile != "v2_split"
                    else False
                )
                table = test[
                    ["date", "session", "symbol", "cut", "end", "y", "z"]
                ].assign(
                    fold=fold,
                    group=group,
                    algorithm=name,
                    score=out,
                    is_probability=probability,
                    alert_line=alert_line,
                    brief_line=brief_line,
                    prior=prior,
                    alert=out >= alert_line,
                    stat_brief=statistical,
                    event_brief=direct,
                    brief=statistical | direct,
                    last_train_end=train.end.max(),
                    inner_train_end=early.end.max(),
                    inner_val_first_cut=validation.cut.min(),
                )
                results.append(table)
                if callback and name == "Ensemble70" and group == "D" and fold == 5:
                    callback(fitted, fields, alert_line, brief_line, prior)
    return pd.concat(results, ignore_index=True)


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


def paired_intervals(pred):
    """All tickers on a date are sampled together in circular five-day blocks."""
    rng = np.random.default_rng(42)
    output = []
    for keys, block in pred.groupby(
        ["clock", "profile", "scope", "target", "session", "algorithm"]
    ):
        if keys[1] not in ["weighted", "filtered"]:
            continue
        wide = block.pivot(
            index=["date", "symbol", "y"], columns="group", values="score"
        ).reset_index()
        dates = list(wide.groupby("date").indices.values())
        n = len(dates)
        y = wide.y.to_numpy()
        predictions = wide[["A", "C", "D"]].to_numpy()
        from scipy.stats import rankdata

        def differences(i):
            yes = y[i].astype(bool)
            npos = yes.sum()
            nneg = len(i) - npos
            ranking = rankdata(predictions[i], axis=0)
            auc = (ranking[yes].sum(axis=0) - npos * (npos + 1) / 2) / (npos * nneg)
            return auc[-1] - auc[:2]

        samples = []
        for _ in range(1000):
            starts = rng.integers(n, size=(n + 4) // 5)
            selected = ((starts[:, None] + np.arange(5)) % n).ravel()[:n]
            i = np.concatenate([dates[j] for j in selected])
            if np.unique(y[i]).size == 2:
                samples.append(differences(i))
        point = differences(np.arange(len(y)))
        lower, upper = np.percentile(samples, [2.5, 97.5], axis=0)
        for j, base in enumerate(["A", "C"]):
            output.append(
                dict(
                    zip(
                        ["clock", "profile", "scope", "target", "session", "algorithm"],
                        keys,
                    ),
                    contrast="D-" + base,
                    delta_auc=point[j],
                    lo95=lower[j],
                    hi95=upper[j],
                )
            )
    return pd.DataFrame(output)
