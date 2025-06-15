# California-Housing-Price-Prediction
California Housing Price Prediction: Advanced ML Pipeline
Project Status: 🚀 Production-Ready

## Overview
This project implements a robust, industry-standard machine learning pipeline for predicting California housing prices using the classic California Housing Dataset. The pipeline features advanced preprocessing, custom feature engineering, systematic model evaluation, and comprehensive diagnostics, demonstrating best practices for real-world ML deployment.

Key Features & Industry Best Practices
Stratified Sampling for Data Splitting

Uses StratifiedShuffleSplit to ensure representative train-test splits, preserving income distribution for unbiased model evaluation.

## Advanced Feature Engineering

Implements custom transformers inheriting from BaseEstimator and TransformerMixin for seamless integration into scikit-learn pipelines.

Engineered features include ratios such as rooms_per_household, population_per_household, and bedrooms_per_room for improved predictive power.

Comprehensive Preprocessing Pipeline

Handles missing values with SimpleImputer.

Scales numerical features with StandardScaler.

Encodes categorical variables with OneHotEncoder.

Manages feature engineering within the pipeline for consistent data flow.

Systematic Model Evaluation

## Implements a 7-step evaluation methodology:

RMSE for error magnitude

MSE for overall performance

R² Score for generalisation assessment

Cross-Validation for robustness and bias detection

Model Comparison for optimal algorithm selection

Residual Analysis for assumption validation and overfitting detection

Feature Importance for interpretability and decision transparency

Professional Debugging & Data Flow Management

Uses print statements and pandas methods for step-by-step data flow inspection.

Handles DataFrame-to-NumPy array conversions and NaN propagation issues for robust preprocessing.

Visual Diagnostics

Generates prediction vs. actual plots, residual histograms, Q-Q plots, and residuals vs. order for comprehensive model validation.

Feature Importance Analysis

Extracts and visualises feature importance for Random Forest models, providing insight into which features drive predictions most strongly.

## Results
Best Model: Random Forest (lowest test RMSE, highest R²)

Model Performance: See results/model_comparison.csv for detailed metrics.

Visual Diagnostics: Check results/evaluation_plots/ for residual analysis and prediction vs. actual plots.

## Industry Highlights
Custom Transformer Architecture: Demonstrates advanced scikit-learn integration with custom BaseEstimator and TransformerMixin classes for feature engineering.

Systematic Evaluation: Implements a 7-step evaluation framework for robust model validation, matching industry standards for production ML.

Data Flow & Debugging: Uses professional debugging techniques and data flow management to ensure pipeline reliability.

Interpretability: Feature importance analysis provides actionable insights for stakeholders and supports model transparency.

Future Work
Hyperparameter Tuning: Grid search and randomised search for further model optimisation.

Deployment: API or web app for real-time predictions.

Monitoring: Model performance tracking and retraining strategies.

Contributing
Contributions are welcome! Please open an issue or submit a pull request.

# License
MIT
