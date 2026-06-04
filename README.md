# Neuro-Symbolic AI for Pasture Biomass Prediction
# AIMS DTU Research Internship 2026

## Overview
This project implements a Logical Tensor Network (LTN) for multi-target pasture biomass regression from 357 RGB images + tabular metadata. The neuro-symbolic approach structurally enforces physical constraints (mass conservation) through a predict-3-derive-2 architecture, achieving perfect physical consistency while maintaining competitive predictive accuracy.

## Key Results (5-fold CV)
| Model | RMSE | R² | Constraint RMSE | Violation Rate |
|---|---|---|---|---|
| XGBoost | 10.13 | 0.753 | 4.33 | 71% |
| LightGBM | 10.25 | 0.749 | 4.79 | 75% |
| Ridge+Poly | 10.63 | 0.722 | 0.004 | 0% |
| Neural-Only | 12.38 | 0.704 | 4.70 | 82% |
| Neural+SoftConstraint | 12.05 | 0.718 | 2.50 | 38% |
| **LTN Neuro-Symbolic** | **10.85** | **0.739** | **0.000** | **0%** |

## Core Innovation
The LTN model predicts only 3 base components (Green, Dead, Clover) and derives the 2 composites (Total, GDM) via exact physical laws, structurally guaranteeing zero constraint violation. Fuzzy logic predicates further enforce domain knowledge (NDVI→Green, Species→Clover, Height monotonicity).

## Structure
- `src/` — Modular codebase (models, training, evaluation, visualization)
- `run_baselines.py` — XGBoost, LightGBM, Ridge baselines
- `run_ltn.py` — LTN neuro-symbolic model training
- `run_evaluation.py` — Statistical tests and visualization
- `outputs/` — Results, figures, and final report
