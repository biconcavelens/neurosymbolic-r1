"""Linear model baselines: Ridge and ElasticNet regression."""
import numpy as np
from typing import Dict
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from .tree_models import _prepare_tabular_data


class RidgeBaseline:
    """Ridge regression with optional polynomial features."""

    def __init__(self, alpha: float = 1.0, use_poly: bool = True, poly_degree: int = 2):
        steps = []
        if use_poly:
            steps.append(("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)))
        steps.extend([
            ("scaler", StandardScaler()),
            ("ridge", MultiOutputRegressor(Ridge(alpha=alpha))),
        ])
        self.model = Pipeline(steps)

    def fit(self, df, preprocessor) -> "RidgeBaseline":
        X, y = _prepare_tabular_data(df, preprocessor)
        self.model.fit(X, y)
        return self

    def predict(self, df, preprocessor) -> np.ndarray:
        X, _ = _prepare_tabular_data(df, preprocessor)
        return self.model.predict(X)


class ElasticNetBaseline:
    """ElasticNet regression baseline."""

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("elasticnet", MultiOutputRegressor(ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000))),
        ])

    def fit(self, df, preprocessor) -> "ElasticNetBaseline":
        X, y = _prepare_tabular_data(df, preprocessor)
        self.model.fit(X, y)
        return self

    def predict(self, df, preprocessor) -> np.ndarray:
        X, _ = _prepare_tabular_data(df, preprocessor)
        return self.model.predict(X)
