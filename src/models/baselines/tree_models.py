"""Tree-based baselines: XGBoost and LightGBM (tabular features only)."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb

TARGET_NAMES = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"]


def _prepare_tabular_data(df: pd.DataFrame, preprocessor) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare tabular features and targets from dataframe."""
    # Aggregate per image
    image_paths = sorted(df["image_path"].unique())
    X_list, y_list = [], []

    for img_path in image_paths:
        img_df = df[df["image_path"] == img_path]
        row = img_df.iloc[0]
        feats = preprocessor.transform(row)
        X_list.append(feats)

        targets = {}
        for _, r in img_df.iterrows():
            targets[r["target_name"]] = r["target"]
        y_list.append([targets[t] for t in TARGET_NAMES])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


class XGBoostBaseline:
    """XGBoost multi-output regression baseline."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 5,
                 learning_rate: float = 0.05, subsample: float = 0.8,
                 colsample_bytree: float = 0.8, random_state: int = 42):
        self.model = MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree, random_state=random_state,
            objective="reg:squarederror", n_jobs=-1,
        ))

    def fit(self, df: pd.DataFrame, preprocessor) -> "XGBoostBaseline":
        X, y = _prepare_tabular_data(df, preprocessor)
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame, preprocessor) -> np.ndarray:
        X, _ = _prepare_tabular_data(df, preprocessor)
        return self.model.predict(X)

    def get_params(self) -> Dict:
        return {"n_estimators": self.model.estimators_[0].n_estimators if hasattr(self.model, 'estimators_') else 300}


class LightGBMBaseline:
    """LightGBM multi-output regression baseline."""

    def __init__(self, n_estimators: int = 300, num_leaves: int = 31,
                 learning_rate: float = 0.05, min_child_samples: int = 10,
                 random_state: int = 42):
        self.model = MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=n_estimators, num_leaves=num_leaves,
            learning_rate=learning_rate, min_child_samples=min_child_samples,
            random_state=random_state, verbose=-1, n_jobs=-1,
        ))

    def fit(self, df: pd.DataFrame, preprocessor) -> "LightGBMBaseline":
        X, y = _prepare_tabular_data(df, preprocessor)
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame, preprocessor) -> np.ndarray:
        X, _ = _prepare_tabular_data(df, preprocessor)
        return self.model.predict(X)

    def get_params(self) -> Dict:
        return {"num_leaves": 31, "n_estimators": 300}
