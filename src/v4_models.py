"""Train-fold-only truncation and separate classification/regression routes."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from analysis_models import folds
from panels import TECH
from risk_evaluation import MARKET
from v3_features import SENTIMENT
from v3_models import TREE_SETTINGS, performance
from v4_features import BASE

WINSOR = ["ClusterImpactSum", "NewsShock", "volume_ratio", "vix_z20", "Interaction"]
MONOTONE = {
    "ClusterImpactSum",
    "NewsShock",
    "HasLevel3Event",
    "is_earnings_window",
    "vix_z20",
}
A = TECH + MARKET + BASE
C = A + SENTIMENT + ["UniqueEventCount", "NoveltyRatio"]
D = C + ["ClusterImpactSum", "MinSentiment", "HasLevel3Event"]
GROUPS = {"A": A, "C": C, "D": D}


class TrainQuantileWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, low=0.01, high=0.99, cols=None):
        self.low = low
        self.high = high
        self.cols = cols

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.positions_ = (
            list(range(values.shape[1]))
            if self.cols is None
            else [X.columns.get_loc(c) for c in self.cols if c in X.columns]
        )
        self.lo_, self.hi_ = np.quantile(
            values[:, self.positions_], [self.low, self.high], axis=0
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "lo_")
        values = np.asarray(X, dtype=float).copy()
        values[:, self.positions_] = np.clip(
            values[:, self.positions_], self.lo_, self.hi_
        )
        return values


def estimators(target, fields):
    if target == "B1":
        linear = LogisticRegression(C=1, max_iter=1500)
        tree = LGBMClassifier(
            monotone_constraints=[int(col in MONOTONE) for col in fields],
            **TREE_SETTINGS,
        )
    else:
        linear = Ridge(alpha=20.0)
        tree = LGBMRegressor(objective="huber", alpha=0.9, **TREE_SETTINGS)
    return (
        make_pipeline(TrainQuantileWinsorizer(cols=WINSOR), StandardScaler(), linear),
        tree,
    )


def fit_components(frame, fields, target):
    value = frame.y if target == "B1" else frame.z.clip(-2.0, 6.0)
    return tuple(
        model.fit(frame[fields], value) for model in estimators(target, fields)
    )


def components(models, frame, fields, target):
    x = frame[fields]
    if target == "B1":
        return np.column_stack([model.predict_proba(x)[:, 1] for model in models])
    return np.column_stack([model.predict(x) for model in models])


def blend(part, model):
    return ((part - model["center"]) / model["scale"]) @ model["weights"]


def fit_route(train, fields, target):
    dates = sorted(train.date.unique())
    gap = dates[int(0.65 * len(dates))]
    val = train[train.date.ge(gap)]
    early = train[(train.date.lt(gap)) & train.end.le(val.cut.min())]
    if target == "B1" and early.y.nunique() != 2:
        raise ValueError("inner training requires both classes")
    valid = components(fit_components(early, fields, target), val, fields, target)
    center = valid.mean(axis=0) if target == "B2" else np.zeros(2)
    scale = valid.std(axis=0) if target == "B2" else np.ones(2)
    scale = np.where(scale < 1e-10, 1, scale)
    route = dict(
        target=target,
        fields=fields,
        models=fit_components(train, fields, target),
        center=center,
        scale=scale,
        weights=np.array([0.7, 0.3] if target == "B1" else [0.5, 0.5]),
        prior=train.y.mean(),
        trained_through=train.end.max(),
        inner_train_end=early.end.max(),
        validation_cut=val.cut.min(),
    )
    route["alert_line"], route["brief_line"] = np.quantile(
        blend(valid, route), [0.85, 0.95]
    )
    return route


def route_scores(frame, route):
    return blend(
        components(route["models"], frame, route["fields"], route["target"]), route
    )


def walk_forward(frame, dates, target, callback=None):
    rows = []
    errors = []
    for fold, days in enumerate(folds(dates, True), 1):
        test = frame[frame.date.isin(days)]
        train = frame[frame.date.lt(min(days)) & frame.end.le(test.cut.min())]
        for name, fields in GROUPS.items():
            route = fit_route(train, fields, target)
            assert route["trained_through"] <= test.cut.min()
            assert route["inner_train_end"] <= route["validation_cut"]
            score = route_scores(test, route)
            prob = target == "B1"
            stat = (score >= route["brief_line"]) | (
                (score >= 3.5 * route["prior"]) if prob else False
            )
            event = test.HasLevel3Event.astype(bool) if name == "D" else False
            rows.append(
                test[
                    [
                        "date",
                        "session",
                        "symbol",
                        "cut",
                        "end",
                        "y",
                        "z",
                        "is_earnings_window",
                        "confirmed_earnings",
                    ]
                ].assign(
                    group=name,
                    fold=fold,
                    score=score,
                    alert_line=route["alert_line"],
                    brief_line=route["brief_line"],
                    last_train_end=route["trained_through"],
                    is_probability=prob,
                    alert=score >= route["alert_line"],
                    stat_brief=stat,
                    event_brief=event,
                    brief=stat | event,
                )
            )
            if name == "D":
                tree = route["models"][1].booster_
                values = tree.predict(test[fields], pred_contrib=True)
                raw = tree.predict(test[fields], raw_score=True)
                error = np.max(np.abs(values.sum(axis=1) - raw))
                assert error < 1e-6
                errors.append(float(error))
                if fold == 5 and callback:
                    callback(route)
    return pd.concat(rows, ignore_index=True), max(errors)


def metrics(table):
    if table.y.nunique() != 2:
        return dict(
            n=len(table), positive=int(table.y.sum()), AUC=np.nan, PR_AUC=np.nan
        )
    return performance(table)


def paired(table, dates, repetitions=1000):
    pivot = table.pivot(
        index=["date", "symbol", "y"], columns="group", values="score"
    ).reset_index()
    groups = pivot.groupby("date").indices
    by_day = [groups.get(date, np.empty(0, dtype=int)) for date in dates]
    truth = pivot.y.to_numpy()
    if len(np.unique(truth)) < 2:
        return [
            dict(
                contrast="D-" + base,
                valid_draws=0,
                delta_auc=np.nan,
                lo95=np.nan,
                hi95=np.nan,
            )
            for base in "AC"
        ]
    values = pivot[["A", "C", "D"]].to_numpy()
    from scipy.stats import rankdata

    def auc_delta(positions):
        label = truth[positions].astype(bool)
        yes = label.sum()
        no = len(label) - yes
        if not yes or not no:
            return None
        ranks = rankdata(values[positions], axis=0)
        auc = (ranks[label].sum(axis=0) - yes * (yes + 1) / 2) / (yes * no)
        return auc[2] - auc[:2]

    rng = np.random.default_rng(20261119)
    samples = []
    for _ in range(repetitions):
        starts = rng.integers(len(dates), size=(len(dates) + 4) // 5)
        draws = ((starts[:, None] + np.arange(5)) % len(dates)).ravel()[: len(dates)]
        sample = auc_delta(np.concatenate([by_day[i] for i in draws]))
        if sample is not None:
            samples.append(sample)
    low, high = np.percentile(samples, [2.5, 97.5], axis=0)
    point = auc_delta(np.arange(len(truth)))
    return [
        dict(
            contrast="D-" + base,
            delta_auc=point[i],
            lo95=low[i],
            hi95=high[i],
            valid_draws=len(samples),
        )
        for i, base in enumerate("AC")
    ]
