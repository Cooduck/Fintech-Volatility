"""Walk-forward classifiers and regressors, with optional signed coefficients."""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from evaluation import folds
from features import BASE, SENTIMENT
from market_data import MARKET, TECH

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

WINSOR = ["ClusterImpactSum", "NewsShock", "volume_ratio", "vix_z20", "Interaction"]
HARD_MONOTONE = {
    "ClusterImpactSum",
    "NewsShock",
    "HasLevel3Event",
    "is_earnings_window",
    "vix_z20",
}
A = TECH + MARKET + BASE
C = A + SENTIMENT + ["UniqueEventCount", "NoveltyRatio"]
D = C + ["ClusterImpactSum", "MinSentiment", "HasLevel3Event"]
HARD_GROUPS = {"A": A, "C": C, "D": D}


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


COLUMNS_C = C + ["NegSentIntensity"]
COLUMNS_D = (
    COLUMNS_C
    + [name for name in D if name not in C and name != "MinSentiment"]
    + ["NegMinIntensity"]
)
GROUPS = {"A": A, "C": COLUMNS_C, "D": COLUMNS_D}
MONOTONE_MAP = dict.fromkeys(
    [
        "ClusterImpactSum",
        "NewsShock",
        "HasLevel3Event",
        "UniqueEventCount",
        "NegRatio",
        "NegSentIntensity",
        "NegMinIntensity",
        "Interaction",
        "is_earnings_window",
        "vix_z20",
    ],
    1,
)


class SoftTailCompression(TrainQuantileWinsorizer):
    """Floor the lower tail and preserve ordering, continuously, above q99."""

    def transform(self, X):
        check_is_fitted(self, "lo_")
        values = np.asarray(X, dtype=float).copy()
        sub = np.maximum(values[:, self.positions_], self.lo_)
        excess = np.maximum(sub - self.hi_, 0.0)
        values[:, self.positions_] = np.minimum(sub, self.hi_) + np.log1p(excess)
        return values


class DirectionalLinear(BaseEstimator):
    """Learn nonnegative risk coefficients while leaving other signs unrestricted."""

    def __init__(self, target, positive):
        self.target = target
        self.positive = positive

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        design = np.column_stack([np.ones(len(values)), values])
        destination = np.asarray(y, dtype=float)
        penalty = 1.0 if self.target == "B1" else 20.0

        def loss(coef):
            signal = design @ coef
            if self.target == "B1":
                cost = np.logaddexp(0, signal) - destination * signal
                residual = expit(signal) - destination
            else:
                residual = signal - destination
                cost = 0.5 * np.square(residual)
            gradient = design.T @ residual
            gradient[1:] += penalty * coef[1:]
            return cost.sum() + 0.5 * penalty * np.square(coef[1:]).sum(), gradient

        restrictions = [(None, None)] + [
            (0, None) if i in self.positive else (None, None)
            for i in range(values.shape[1])
        ]
        optimum = minimize(
            loss,
            np.zeros(design.shape[1]),
            jac=True,
            method="L-BFGS-B",
            bounds=restrictions,
            options={"maxiter": 4000, "ftol": 1e-12},
        )
        if not optimum.success:
            raise RuntimeError(optimum.message)
        self.intercept_ = float(optimum.x[0])
        self.coef_ = optimum.x[1:]
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return self.intercept_ + np.asarray(X) @ self.coef_

    def predict_proba(self, X):
        p = expit(self.predict(X))
        return np.column_stack([1 - p, p])


def make_linear(target, fields, candidate):
    if candidate == "signed":
        model = DirectionalLinear(
            target, tuple(i for i, name in enumerate(fields) if name in MONOTONE_MAP)
        )
    else:
        model = (
            LogisticRegression(C=1, max_iter=1500)
            if target == "B1"
            else Ridge(alpha=20.0)
        )
    transformer = TrainQuantileWinsorizer if candidate == "hard" else SoftTailCompression
    return make_pipeline(transformer(cols=WINSOR), StandardScaler(), model)


def make_tree(target, fields, candidate):
    config = dict(TREE_SETTINGS)
    # The historical hard-clipped control left its regression tree unrestricted.
    if candidate != "hard":
        config["monotone_constraints"] = [MONOTONE_MAP.get(field, 0) for field in fields]
    elif target == "B1":
        config["monotone_constraints"] = [int(field in HARD_MONOTONE) for field in fields]
    if target == "B1":
        return LGBMClassifier(**config)
    return LGBMRegressor(objective="huber", alpha=0.9, **config)


def components(models, frame, fields, target, linear_fields=None):
    x = frame[fields]
    linear = frame[fields if linear_fields is None else linear_fields]
    if target == "B1":
        return np.column_stack(
            [models[0].predict_proba(linear)[:, 1], models[1].predict_proba(x)[:, 1]]
        )
    return np.column_stack([models[0].predict(linear), models[1].predict(x)])


def blend(part, model):
    return ((part - model["center"]) / model["scale"]) @ model["weights"]


def fit_route(train, fields, target, candidate="soft"):
    if candidate not in {"hard", "soft", "signed"}:
        raise ValueError(f"Unknown model candidate: {candidate}")
    dates = sorted(train.date.unique())
    if len(dates) < 2:
        raise ValueError("Chronological training requires at least two sessions.")
    gap = dates[int(0.65 * len(dates))]
    val = train[train.date.ge(gap)]
    early = train[train.date.lt(gap) & train.end.le(val.cut.min())]

    if target == "B1" and early.y.nunique() != 2:
        raise ValueError("Inner training requires both classes.")

    def response(batch):
        return batch.y if target == "B1" else batch.z.clip(-2.0, 6.0)

    linear_fields = list(fields)
    early_linear = make_linear(target, linear_fields, candidate).fit(
        early[linear_fields], response(early)
    )
    comparison = {
        "novelty_inner_coef": np.nan,
        "novelty_inner_auc_delta": np.nan,
        "novelty_kept": False,
    }
    if candidate != "hard" and target == "B1" and "NoveltyRatio" in fields:
        weights = np.ravel(early_linear[-1].coef_)
        coefficient = weights[linear_fields.index("NoveltyRatio")]
        comparison.update(novelty_inner_coef=coefficient, novelty_kept=True)
        if coefficient < 0 and val.y.nunique() == 2:
            reduced = [f for f in fields if f != "NoveltyRatio"]
            alternative = make_linear(target, reduced, candidate).fit(
                early[reduced], response(early)
            )
            with_novelty = roc_auc_score(
                val.y, early_linear.predict_proba(val[linear_fields])[:, 1]
            )
            without_novelty = roc_auc_score(
                val.y, alternative.predict_proba(val[reduced])[:, 1]
            )
            comparison["novelty_inner_auc_delta"] = with_novelty - without_novelty
            if with_novelty <= without_novelty:
                linear_fields = reduced
                early_linear = alternative
                comparison["novelty_kept"] = False
    early_tree = make_tree(target, fields, candidate).fit(early[fields], response(early))
    valid = components((early_linear, early_tree), val, fields, target, linear_fields)
    center = valid.mean(axis=0) if target == "B2" else np.zeros(2)
    scale = valid.std(axis=0) if target == "B2" else np.ones(2)
    scale = np.where(scale < 1e-10, 1, scale)
    model = dict(
        target=target,
        fields=fields,
        linear_fields=linear_fields,
        candidate=candidate,
        models=(
            make_linear(target, linear_fields, candidate).fit(
                train[linear_fields], response(train)
            ),
            make_tree(target, fields, candidate).fit(train[fields], response(train)),
        ),
        center=center,
        scale=scale,
        weights=np.array([0.7, 0.3] if target == "B1" else [0.5, 0.5]),
        prior=train.y.mean(),
        trained_through=train.end.max(),
        inner_train_end=early.end.max(),
        validation_cut=val.cut.min(),
        **comparison,
    )
    model["alert_line"], model["brief_line"] = np.quantile(
        blend(valid, model), [0.85, 0.95]
    )
    return model


def fit_hard_route(train, fields, target):
    """Reproduce the published hard-clipped benchmark with the shared fitter."""
    return fit_route(train, fields, target, candidate="hard")


def route_scores(frame, route):
    return blend(
        components(
            route["models"],
            frame,
            route["fields"],
            route["target"],
            route.get("linear_fields"),
        ),
        route,
    )


def walk_forward(
    frame,
    dates,
    target,
    callback=None,
    *,
    groups=GROUPS,
    route_fitter=fit_route,
    audit=None,
):
    rows = []
    errors = []
    for fold, days in enumerate(folds(dates), 1):
        test = frame[frame.date.isin(days)]
        train = frame[frame.date.lt(min(days)) & frame.end.le(test.cut.min())]
        for name, fields in groups.items():
            route = route_fitter(train, fields, target)
            if audit:
                audit(route, fold, name)
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


def weights(route, fold, group):
    return [
        dict(
            fold=fold,
            group=group,
            feature=name,
            coefficient=float(coef),
            novelty_inner_coef=route["novelty_inner_coef"],
            novelty_inner_auc_delta=route["novelty_inner_auc_delta"],
            novelty_kept=route["novelty_kept"],
            validation_cut=route["validation_cut"],
        )
        for name, coef in zip(
            route["linear_fields"], np.ravel(route["models"][0][-1].coef_)
        )
    ]
