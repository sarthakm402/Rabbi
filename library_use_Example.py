import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split

# Load your library functions (assuming they are in a script called `library_ML.py`)
from library_ML import (
    detect_anomaly, missing_values, scale, preprocess_features, 
    regression_model, classification_model
)
  
# ✅ 1. Load Sample Dataset 
df = pd.DataFrame(np.random.randn(100, 5), columns=[f"Feature_{i}" for i in range(5)])  # Fake dataset

# ✅ 2. Check for Anomalies
print("Before Anomaly Detection:")
print(df.describe())
detect_anomaly(df)
print("After Anomaly Detection:")
print(df.describe())

# ✅ 3. Handle Missing Values
df = missing_values(df, impute=True, strategy="mean", drop_threshold=0.5)

# ✅ 4. Scale Data   
scaled_data = scale(df, standard=True)

# ✅ 5. Feature Selection (Variance & Correlation)
filtered_data = preprocess_features(df, variance=True, correlation=True)

# ✅ 6. Regression Model Testing (Using California Housing Dataset)
california = fetch_california_housing()
X_reg, y_reg = california.data, california.target
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

print("Training Regression Model...")
reg_results = regression_model(X_reg_train, y_reg_train, models=['linear', 'random_forest', 'xgboost'])

# ✅ 7. Classification Model Testing (Using Iris Dataset)
iris = load_iris()
X_clf, y_clf = iris.data, iris.target
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

print("Training Classification Model...")
clf_results = classification_model(X_clf_train, y_clf_train, models=['logistic', 'random_forest', 'xgboost'])

# ✅ 8. Print Results
print("\nRegression Model Results:")
for model, result in reg_results.items():
    print(f"{model}: MSE={result['mse']}, MAE={result['mae']}, R2={result['r2']}")

print("\nClassification Model Results:")
for model, result in clf_results.items():
    print(f"{model}: Accuracy={result['accuracy']}, F1 Score={result['f1_score']}")

