"""Chronological probability calibration, alert policies and pooled model comparisons."""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
)
from lightgbm import LGBMClassifier
from panels import TECH, SEM
from analysis_models import folds

MARKET = ["qqq_ret1", "qqq_vol5", "qqq_vol20", "rel_vol_ratio"]


def estimator(name, balanced):
    weight = "balanced" if balanced else None
    if name == "L2":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1, max_iter=1500, class_weight=weight),
        )
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=90,
            min_samples_leaf=6,
            max_features=0.8,
            class_weight=weight,
            n_jobs=2,
            random_state=24,
        )
    return LGBMClassifier(
        n_estimators=60,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=5,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=24,
        n_jobs=2,
        verbosity=-1,
        class_weight=weight,
    )


def calibrations(v, y, p):
    yield "raw", v, p
    iso = IsotonicRegression(out_of_bounds="clip").fit(v, y)
    yield "isotonic", iso.predict(v), iso.predict(p)

    def logit(x):
        x = np.clip(x, 1e-6, 1 - 1e-6)
        return np.log(x / (1 - x)).reshape(-1, 1)

    platt = LogisticRegression(C=1).fit(logit(v), y)
    yield (
        "platt",
        platt.predict_proba(logit(v))[:, 1],
        platt.predict_proba(logit(p))[:, 1],
    )


def policy(v, y, prior):
    precision, recall, thresholds = precision_recall_curve(y, v)

    def best(beta):
        f = (
            (1 + beta**2)
            * precision[:-1]
            * recall[:-1]
            / np.maximum(beta**2 * precision[:-1] + recall[:-1], 1e-12)
        )
        return thresholds[np.flatnonzero(f == max(f))[-1]]

    return dict(
        alert_line=np.quantile(v, 0.85),
        brief_line=np.quantile(v, 0.95),
        f1_line=best(1),
        f2_line=best(2),
        prior=prior,
    )


def evaluate(frame, dates, case_callback=None, recent=True):
    tech = (
        TECH
        + MARKET
        + (["is_afternoon_session"] if frame.session.nunique() > 1 else [])
    )
    groups = {
        "A": tech,
        "B": tech + ["Lexicon"],
        "C": tech + SEM[:6] + ["has_news"],
        "D": tech + SEM + ["has_news", "max_impact_level"],
    }
    parts = []
    for fold, test_dates in enumerate(folds(dates, recent), 1):
        test = frame[frame.date.isin(test_dates)]
        train = frame[(frame.date < min(test_dates)) & (frame.end <= test.cut.min())]
        days = sorted(train.date.unique())
        split = days[int(0.65 * len(days))]
        val = train[train.date >= split]
        early = train[(train.date < split) & (train.end <= val.cut.min())]
        assert early.end.max() <= val.cut.min() and val.end.max() <= test.cut.min()
        assert val.y.nunique() == 2 and early.y.nunique() == 2
        for group, cols in groups.items():
            for balanced in [False, True] if group == "D" else [False]:
                preds = {}
                for name in ["L2", "LightGBM", "RF"]:
                    base = estimator(name, balanced).fit(early[cols], early.y)
                    v = base.predict_proba(val[cols])[:, 1]
                    model = estimator(name, balanced).fit(train[cols], train.y)
                    p = model.predict_proba(test[cols])[:, 1]
                    preds[name] = (v, p)
                    if (
                        name == "LightGBM"
                        and group == "D"
                        and not balanced
                        and case_callback
                    ):
                        case_callback(
                            fold,
                            model,
                            cols,
                            train,
                            test,
                            p,
                            policy(v, val.y, train.y.mean()),
                        )
                if group == "D":
                    preds["Ensemble"] = tuple(
                        0.5 * (a + b) for a, b in zip(preds["L2"], preds["LightGBM"])
                    )
                for name, (v, p) in preds.items():
                    variants = (
                        calibrations(v, val.y, p) if group == "D" else [("raw", v, p)]
                    )
                    for method, valid, out in variants:
                        spec = policy(valid, val.y, train.y.mean())
                        obs = test[
                            ["date", "session", "symbol", "cut", "end", "y"]
                        ].copy()
                        obs = obs.assign(
                            fold=fold,
                            group="E" if name == "Ensemble" else group,
                            algorithm=name,
                            balanced=balanced,
                            calibration=method,
                            p=out,
                            **spec,
                            last_train_end=train.end.max(),
                            inner_train_end=early.end.max(),
                            inner_val_first_cut=val.cut.min(),
                        )
                        obs["alert"] = obs.p >= obs.alert_line
                        obs["brief"] = (obs.p >= obs.brief_line) | (
                            obs.p / obs.prior >= 3.5
                        )
                        parts.append(obs)
    return pd.concat(parts, ignore_index=True)


def performance(d):
    y = d.y.astype(bool).to_numpy()
    p = d.p.to_numpy()
    a = d.alert.to_numpy()
    rate = y.mean()
    tp = (y & a).sum()

    def recall(m):
        return (y & m).sum() / max(y.sum(), 1)

    def f1(m):
        return 2 * (y & m).sum() / max(y.sum() + m.sum(), 1)

    # Ex-post monthly ranking diagnostic. Date/ticker/session order resolves ties.
    ranked = d.sort_values(["p", "cut", "symbol"], ascending=[False, True, True]).copy()
    by = ranked.groupby(ranked.date.str[:7])
    rank = by.cumcount()
    n = by.y.transform("size")
    top15 = ranked[rank < np.floor(0.15 * n)]
    top5 = ranked[rank < np.floor(0.05 * n)]
    return dict(
        n=len(y),
        positive=int(sum(y)),
        AUC=roc_auc_score(y, p) if 0 < rate < 1 else np.nan,
        PR_AUC=average_precision_score(y, p),
        Brier=brier_score_loss(y, p),
        Recall=recall(a),
        Precision=tp / max(sum(a), 1),
        F1=f1(a),
        alert_rate=a.mean(),
        alerts=int(sum(a)),
        briefs=int(d.brief.sum()),
        Recall_Top15=top15.y.sum() / max(y.sum(), 1),
        Lift_Top5=top5.y.mean() / rate if rate else np.nan,
        F1_at_f1=f1(p >= d.f1_line.to_numpy()),
        Recall_at_f2=recall(p >= d.f2_line.to_numpy()),
        F1_at_f2=f1(p >= d.f2_line.to_numpy()),
    )
