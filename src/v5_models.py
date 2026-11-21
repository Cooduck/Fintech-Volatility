"""Final-round tail compression, sign constraints and chronological feature audit."""

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from v3_models import TREE_SETTINGS
from v4_models import A, C, D, WINSOR, TrainQuantileWinsorizer, blend, components

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


def derive(frame):
    return frame.assign(
        NegSentIntensity=(-frame.SentMean).clip(lower=0),
        NegMinIntensity=(-frame.MinSentiment).clip(lower=0),
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
    return make_pipeline(SoftTailCompression(cols=WINSOR), StandardScaler(), model)


def make_tree(target, fields):
    config = dict(
        TREE_SETTINGS, monotone_constraints=[MONOTONE_MAP.get(f, 0) for f in fields]
    )
    return (
        LGBMClassifier(**config)
        if target == "B1"
        else LGBMRegressor(objective="huber", alpha=0.9, **config)
    )


def fit_route(train, fields, target, candidate="soft"):
    dates = sorted(train.date.unique())
    gap = dates[int(0.65 * len(dates))]
    val = train[train.date.ge(gap)]
    early = train[train.date.lt(gap) & train.end.le(val.cut.min())]

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
    if target == "B1" and "NoveltyRatio" in fields:
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
    early_tree = make_tree(target, fields).fit(early[fields], response(early))
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
            make_tree(target, fields).fit(train[fields], response(train)),
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
