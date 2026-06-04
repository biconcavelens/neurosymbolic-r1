"""Centralized configuration for the Neuro-Symbolic biomass prediction project."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "train"
CSV_PATH = DATA_DIR / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"
PRED_DIR = OUTPUT_DIR / "predictions"
LOG_DIR = OUTPUT_DIR / "logs"

for d in [MODEL_DIR, FIGURE_DIR, PRED_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    image_size: Tuple[int, int] = (512, 256)  # (width, height), preserves 2:1 aspect ratio
    grid_rows: int = 4
    grid_cols: int = 2
    use_grid_features: bool = True
    use_color_histogram: bool = True
    color_hist_bins: int = 64

    # Target transforms
    log1p_targets: List[str] = field(default_factory=lambda: ["Dry_Total_g", "Dry_Green_g", "GDM_g"])
    sqrt_targets: List[str] = field(default_factory=lambda: ["Dry_Dead_g"])
    two_stage_targets: List[str] = field(default_factory=lambda: ["Dry_Clover_g"])

    # Tabular preprocessing
    state_embedding_dim: int = 4
    species_embedding_dim: int = 8
    date_cyclical: bool = True

    # Normalization stats (computed from data)
    ndvi_mean: float = 0.6574
    ndvi_std: float = 0.1520
    height_mean: float = 7.596
    height_std: float = 10.274
    height_log_mean: float = 1.589
    height_log_std: float = 0.703


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Image encoder
    backbone: str = "efficientnet_b0"  # or resnet18
    pretrained: bool = True
    image_feature_dim: int = 256
    backbone_freeze_epochs: int = 10

    # Tabular encoder
    tabular_hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    tabular_dropout: float = 0.2

    # Fusion
    fusion_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    fusion_dropout: float = 0.3

    # Prediction heads
    prediction_head_dims: List[int] = field(default_factory=lambda: [64, 32])

    # Grid feature encoder (smaller CNN)
    grid_backbone: str = "efficientnet_b0"

    # Output targets
    base_targets: List[str] = field(default_factory=lambda: ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"])
    derived_targets: List[str] = field(default_factory=lambda: ["Dry_Total_g", "GDM_g"])
    all_targets: List[str] = field(default_factory=lambda: [
        "Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "Dry_Total_g", "GDM_g"
    ])


@dataclass
class LTNConfig:
    """Logical Tensor Network configuration."""
    # Constraint loss weights (3-phase schedule)
    phase1_epochs: int = 10
    phase2_epochs: int = 25
    phase1_weights: Tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.0)
    phase2_weights: Tuple[float, float, float, float] = (1.0, 1.5, 0.5, 0.2)
    phase3_weights: Tuple[float, float, float, float] = (1.0, 2.0, 1.0, 0.5)

    # Predicate parameters
    constraint_eps: float = 0.5  # Tolerance for fuzzy equality
    ndvi_threshold_init: float = 0.6  # Initial threshold for "high NDVI"
    height_threshold_init: float = 5.0  # Initial threshold for "tall pasture"

    # Predicate types
    use_learned_weights: bool = True  # Per-sample predicate weight gating
    use_learned_predicates: bool = True  # Differentiable rule discovery
    use_evidential: bool = True  # Evidential regression heads

    # t-norm choice
    t_norm: str = "lukasiewicz"  # or "product", "godel"

    # Aggregation
    p_norm: int = 2  # For p-mean error aggregation


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # General
    batch_size: int = 16
    num_epochs: int = 60
    num_workers: int = 0  # 0 = main process only (safest on Windows)
    seed: int = 42

    # Optimizer
    lr_backbone: float = 1e-4
    lr_heads: float = 1e-3
    weight_decay: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    lr_scheduler: str = "cosine"  # cosine, plateau, step
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_rmse_total"

    # Cross-validation
    num_folds: int = 5
    val_split: float = 0.2

    # Contrastive pre-training
    contrastive_epochs: int = 30
    contrastive_temperature: float = 0.1
    contrastive_projection_dim: int = 128

    # Augmentation
    use_train_augmentation: bool = True
    use_tta: bool = True  # Test-time augmentation
    tta_views: int = 5

    # Mix precision
    use_amp: bool = True  # Automatic mixed precision


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ltn: LTNConfig = field(default_factory=LTNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "neuro_symbolic_biomass"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def run_dir(self) -> Path:
        return OUTPUT_DIR

    @property
    def model_dir(self) -> Path:
        return MODEL_DIR

    @property
    def figure_dir(self) -> Path:
        return FIGURE_DIR


# Default config instance
default_config = ExperimentConfig()
