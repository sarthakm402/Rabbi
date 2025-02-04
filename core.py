import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, PowerTransformer, FunctionTransformer
# Feature Engineering
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import SVR
import optuna
from sklearn.model_selection import cross_val_score, train_test_split
# Model Evaluation
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
)

def detect_anomaly(df):
    """
    Detects anomalies in a dataset using the Interquartile Range (IQR) method.

    This function identifies outliers in each column of the dataframe by calculating the
    IQR and then replacing values outside the lower and upper bounds with NaN.

    Parameters:
    - df (pd.DataFrame): The dataset to process. Each column is evaluated for outliers.
    
    Returns:
    - pd.DataFrame: A modified copy of the dataset with anomalies (outliers) replaced by NaN.
    """
    df_copy = df.copy()  # Create a copy of the original dataframe
    for i in df_copy.columns:
        Q1 = df_copy[i].quantile(0.25)
        Q3 = df_copy[i].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        df_copy.loc[(df_copy[i] < lower_bound) | (df_copy[i] > upper_bound), i] = np.nan
        print(f"Anomalies detected in column {i}")
    return df_copy  # Return the modified copy of the dataframe


def missing_values(data, impute=True, strategy="mean", drop_threshold=None):
    """
    Handles missing values in a dataset.

    Parameters:
    - data (pd.DataFrame): The dataset to process.
    - impute (bool): Whether to impute missing values.
    - strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent').
    - drop_threshold (float): If specified, drops columns with missingness above this threshold.

    Returns:
    - pd.DataFrame: The processed dataset (modified copy).
    """
    data_copy = data.copy()  # Create a copy of the original dataframe
    if drop_threshold is not None:
        missing_percentages = data_copy.isnull().mean()
        to_drop = missing_percentages[missing_percentages > drop_threshold].index
        data_copy = data_copy.drop(columns=to_drop)
        print(f"Dropped columns: {list(to_drop)}")
    
    if impute:
        imputer = SimpleImputer(strategy=strategy)
        data_copy = pd.DataFrame(imputer.fit_transform(data_copy), columns=data_copy.columns)
        print(f"Imputed missing values using strategy: {strategy}")
    
    return data_copy  # Return the modified copy


def scale(data, standard=True, min_max=False, power_transform=False, one_hot=False, power_method='yeo-johnson', log_transform=False, categories='auto'):
    """
    Scales or encodes the input data using various preprocessing techniques based on the specified parameters.

    Parameters:
    - data (pd.DataFrame or np.ndarray): The dataset to preprocess.
    - standard (bool): If True, applies StandardScaler.
    - min_max (bool): If True, applies MinMaxScaler.
    - power_transform (bool): If True, applies PowerTransformer.
    - one_hot (bool): If True, applies OneHotEncoder.
    - power_method (str): Method for PowerTransformer ('yeo-johnson' or 'box-cox').
    - log_transform (bool): If True, applies log1p transformation.
    - categories (str or list of lists): Used by OneHotEncoder to specify how categories are handled.

    Returns:
    - np.ndarray or pd.DataFrame: The preprocessed dataset.
    """
    if standard:
        scaler = StandardScaler()
        data_transformed = scaler.fit_transform(data)
    elif min_max:
        scaler = MinMaxScaler()
        data_transformed = scaler.fit_transform(data)
    elif power_transform:
        transformer = PowerTransformer(method=power_method)
        data_transformed = transformer.fit_transform(data)
    elif log_transform:
        transformer = FunctionTransformer(func=np.log1p, validate=True)
        data_transformed = transformer.fit_transform(data)
    elif one_hot:
        encoder = OneHotEncoder(categories=categories, sparse=False)
        data_transformed = encoder.fit_transform(data)
    else:
        raise ValueError("At least one transformation method (s, m, p, log_transform, or o) must be set to True.")
    
    return data_transformed  # Return the transformed data


def preprocess_features(data, variance=False, correlation=False, variance_threshold=0.01, correlation_threshold=0.9):
    """
    Removes low-variance features and highly correlated features from the dataset.

    Parameters:
    - data (pd.DataFrame): The input dataset.
    - variance_threshold (float): Minimum variance required for a feature to be retained.
    - correlation_threshold (float): Correlation threshold above which features are dropped.

    Returns:
    - pd.DataFrame: A modified copy of the dataset with low-variance and highly correlated features removed.
    """
    data_copy = data.copy()  # Create a copy of the original dataframe
    if variance:
        selector = VarianceThreshold(threshold=variance_threshold)
        low_variance_data = selector.fit_transform(data_copy)
        retained_features = data_copy.columns[selector.get_support()]
        data_copy = pd.DataFrame(low_variance_data, columns=retained_features)

    if correlation:
        corr_matrix = data_copy.corr()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column].abs() > correlation_threshold)]
        data_copy = data_copy.drop(columns=to_drop)

    return data_copy  # Return the modified copy


def regression_model(train, labels, models=['linear', 'random_forest', 'xgboost', 'svr'], n_trials=50):
    """
    Trains multiple regression models with hyperparameter tuning using Optuna and evaluates their performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.3, random_state=42)

    results = {}
    if isinstance(models, str):
        models = [models]

    def objective(trial, model_type):
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            model = RandomForestRegressor(**params, random_state=42)
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = XGBRegressor(**params, random_state=42)
        elif model_type == 'svr':
            params = {
                'C': trial.suggest_float('C', 0.1, 10),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            }
            model = SVR(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        return -np.mean(scores)

    for model_type in models:
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results['linear'] = {'model': model, 'params': None, 'mse': mse, 'mae': mae, 'r2': r2, 'predictions': predictions}
        else:
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective(trial, model_type), n_trials=n_trials)
            best_params = study.best_params
            
            if model_type == 'random_forest':
                best_model = RandomForestRegressor(**best_params, random_state=42)
            elif model_type == 'xgboost':
                best_model = XGBRegressor(**best_params, random_state=42)
            elif model_type == 'svr':
                best_model = SVR(**best_params)

            best_model.fit(X_train, y_train)
            predictions = best_model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results[model_type] = {'model': best_model, 'params': best_params, 'mse': mse, 'mae': mae, 'r2': r2, 'predictions': predictions}
    print(results)
    return results


def classification_model(train, labels, models=['logistic', 'random_forest', 'xgboost'], n_trials=50):
    """
    Trains multiple classification models with hyperparameter tuning using Optuna and evaluates their performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.3, random_state=42)

    results = {}
    if isinstance(models, str):
        models = [models]

    def objective(trial, model_type):
        if model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
            model = RandomForestClassifier(**params, random_state=42)
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            model = XGBClassifier(**params, random_state=42)
        elif model_type == 'logistic':
            params = {
                'C': trial.suggest_float('C', 0.1, 10),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
            }
            model = LogisticRegression(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        f1 = f1_score(y_test, predictions, average='weighted')
        return f1

    for model_type in models:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_type), n_trials=n_trials)
        best_params = study.best_params
        
        if model_type == 'random_forest':
            best_model = RandomForestClassifier(**best_params, random_state=42)
        elif model_type == 'xgboost':
            best_model = XGBClassifier(**best_params, random_state=42)
        elif model_type == 'logistic':
            best_model = LogisticRegression(**best_params)

        best_model.fit(X_train, y_train)
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')
        results[model_type] = {'model': best_model, 'params': best_params, 'accuracy': accuracy, 'f1_score': f1, 'predictions': predictions}
    
    print(results)
    return results
