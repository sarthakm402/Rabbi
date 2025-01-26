# Machine Learning Feature Engineering and Modeling Toolkit

## Overview
This repository contains a comprehensive set of tools and functions for data preprocessing, feature engineering, and model training. It supports both regression and classification tasks with hyperparameter tuning using Optuna, ensuring efficient and accurate results.

---

## Features
1. **Anomaly Detection**: Detects outliers using the Interquartile Range (IQR) method and replaces them with `NaN`.
2. **Missing Value Handling**: Imputes or drops missing values with customizable strategies (`mean`, `median`, etc.).
3. **Scaling and Transformation**: Provides multiple scaling methods (StandardScaler, MinMaxScaler), power transformations (Yeo-Johnson, Box-Cox), and log transformations.
4. **Feature Engineering**:  
   - Removes low-variance features.  
   - Eliminates highly correlated features based on thresholds.
5. **Regression Models**: Supports Linear Regression, Random Forest, XGBoost, and SVR with hyperparameter tuning.
6. **Classification Models**: Includes Logistic Regression, Random Forest, and XGBoost with hyperparameter optimization.
7. **Customizable Parameters**: Allows easy configuration for preprocessing, modeling, and evaluation.

---

## Usage

### 1. Anomaly Detection  
Detect anomalies in the dataset using the Interquartile Range (IQR) method and replace them with `NaN`.


### 2. Handle Missing Values
Impute or drop missing values from the dataset using the specified strategy (e.g., mean, median, etc.).

### 3. Scaling and Transformation
Apply various scaling methods and transformations (e.g., StandardScaler, MinMaxScaler, PowerTransformer, etc.).

### 4. Feature Engineering
Remove low-variance features and highly correlated features from the dataset.

### 5. Regression Modeling
Train and evaluate multiple regression models with hyperparameter tuning using Optuna.

### 6. Classification Modeling
Train and evaluate multiple classification models with hyperparameter optimization using Optuna.

## Dependencies
This project requires the following Python libraries:

- pandas
- numpy
- scikit-learn
- seaborn
- xgboost
- optuna

