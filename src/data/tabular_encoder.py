"""Preprocessing for tabular features: state, species, NDVI, height, date."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import torch
import torch.nn as nn


class TabularPreprocessor:
    """Fit tabular feature encoders and transform raw CSV data into tensors."""

    def __init__(self, state_embedding_dim: int = 4, species_embedding_dim: int = 8,
                 date_cyclical: bool = True):
        self.state_embedding_dim = state_embedding_dim
        self.species_embedding_dim = species_embedding_dim
        self.date_cyclical = date_cyclical

        # Will be populated by fit()
        self.state_to_idx: Dict[str, int] = {}
        self.species_to_idx: Dict[str, int] = {}
        self.ndvi_mean: float = 0.0
        self.ndvi_std: float = 0.0
        self.height_mean: float = 0.0
        self.height_std: float = 0.0
        self.height_log_mean: float = 0.0
        self.height_log_std: float = 0.0
        self.num_state: int = 0
        self.num_species: int = 0
        self.output_dim: int = 0

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit encoders to the dataframe."""
        self.state_to_idx = {s: i for i, s in enumerate(sorted(df["State"].unique()))}
        self.species_to_idx = {s: i for i, s in enumerate(sorted(df["Species"].unique()))}
        self.num_state = len(self.state_to_idx)
        self.num_species = len(self.species_to_idx)

        self.ndvi_mean = float(df["Pre_GSHH_NDVI"].mean())
        self.ndvi_std = float(df["Pre_GSHH_NDVI"].std())
        self.height_mean = float(df["Height_Ave_cm"].mean())
        self.height_std = float(df["Height_Ave_cm"].std())

        log_height = np.log1p(df["Height_Ave_cm"].values)
        self.height_log_mean = float(log_height.mean())
        self.height_log_std = float(log_height.std())

        # Output dimension: ndvi(1) + log_height(1) + state_embed + species_embed + date_cyclical(2)
        self.output_dim = 1 + 1 + self.num_state + self.num_species + (2 if self.date_cyclical else 1)
        return self

    def transform(self, row: pd.Series) -> np.ndarray:
        """Transform a single row into a feature vector."""
        features = []

        # NDVI (standardize)
        ndvi = (row["Pre_GSHH_NDVI"] - self.ndvi_mean) / (self.ndvi_std + 1e-8)
        features.append(ndvi)

        # Height (log-transform then standardize)
        log_h = np.log1p(row["Height_Ave_cm"])
        log_h_std = (log_h - self.height_log_mean) / (self.height_log_std + 1e-8)
        features.append(log_h_std)

        # State (one-hot)
        state_idx = self.state_to_idx.get(row["State"], 0)
        state_onehot = np.zeros(self.num_state, dtype=np.float32)
        state_onehot[state_idx] = 1.0
        features.extend(state_onehot)

        # Species (one-hot)
        species_idx = self.species_to_idx.get(row["Species"], 0)
        species_onehot = np.zeros(self.num_species, dtype=np.float32)
        species_onehot[species_idx] = 1.0
        features.extend(species_onehot)

        # Date (cyclical encoding)
        if self.date_cyclical:
            date_str = row["Sampling_Date"]
            parts = date_str.split("/")
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            from datetime import date
            doy = date(year, month, day).timetuple().tm_yday
            features.append(np.sin(2 * np.pi * doy / 365))
            features.append(np.cos(2 * np.pi * doy / 365))

        return np.array(features, dtype=np.float32)

    def transform_batch(self, df: pd.DataFrame) -> torch.Tensor:
        """Transform a dataframe of rows into a tensor."""
        feats = np.stack([self.transform(row) for _, row in df.iterrows()])
        return torch.tensor(feats, dtype=torch.float32)


class TabularEncoder(nn.Module):
    """Neural network encoder for tabular features."""

    def __init__(self, input_dim: int, hidden_dims=(64, 32, 16), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = hdim
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
