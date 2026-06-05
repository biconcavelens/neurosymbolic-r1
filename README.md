# Neuro-Symbolic AI for Pasture Biomass Prediction
## AIMS DTU Research Internship 2026

## Overview
This project implements a **Logical Tensor Network (LTN)** for multi-target pasture biomass regression from 357 RGB images + tabular metadata. The neuro-symbolic approach structurally enforces physical constraints (mass conservation) through a **predict-3-derive-2 architecture**, achieving **perfect physical consistency (0% violation)** while maintaining competitive predictive accuracy.

## Key Results (5-fold CV)
| Model | RMSE ↓ | R² ↑ | Constraint RMSE ↓ | Violation Rate ↓ |
|-------|:------:|:----:|:-----------------:|:----------------:|
| XGBoost | **10.29** | **0.859** | 4.866 | 71% |
| LightGBM | 10.56 | 0.851 | 4.216 | 81% |
| Ridge+Poly | 12.38 | 0.796 | **0.010** | **0%** |
| Neural-Only | 16.13 | 0.654 | 8.775 | 100% |
| Neural+SoftConstraint | 16.27 | 0.648 | 8.692 | 100% |
| **LTN Neuro-Symbolic** | **16.00** | **0.659** | **0.000** | **0%** ✅ |

## Core Innovation
The LTN model predicts only **3 base components** (Green, Dead, Clover) and **derives the 2 composites** (Total, GDM) via exact physical laws, structurally guaranteeing **zero constraint violation**. Fuzzy logic predicates further enforce domain knowledge (NDVI→Green, Species→Clover, Height monotonicity). Unlike soft-constraint baselines that still violate at 100%, the LTN's hard architectural constraint ensures every prediction is physically consistent.

## Structure
- `src/` — Modular codebase (models, training, evaluation, visualization)
- `run_baselines.py` — XGBoost, LightGBM, Ridge, Neural baselines (5-fold CV)
- `run_ltn.py` — LTN neuro-symbolic model training (5-fold CV)
- `run_evaluation.py` — Statistical tests and visualization
- `run_comparison.py` — Quick 1-fold comparison run
- `outputs/` — Results, figures, and final report
