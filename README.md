# Auto-retraining Gradient Boosting Pipeline  
with MLflow + Prefect

## Overview

This project implements an **end-to-end pipeline** for periodic retraining of a 
gradient boosting model on fresh data, with automatic experiment tracking, metric comparison,
and promotion of better models to production.

Key features:
![prefect.jpg](static/prefect.jpg)
- Baseline model training with **Optuna** hyperparameter & feature tuning (optional)
- Scheduled / triggered retraining on new data
- **Data drift detection** using **Evidently**
- Every run is logged to **MLflow**
- Automatic comparison of the new (challenger) model vs current champion
- If challenger performs better → it becomes the new production model
- Full orchestration done with **Prefect 2**