import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt
import os

def prepare_data(df_path: str, dvc_file: str = '../data/cleaned_data.csv.dvc') -> tuple:
    """
    Prepare data for modeling by handling missing values, feature engineering, encoding, and splitting.

    Returns:
        tuple: Training and test splits for severity and probability models.
    """
    os.system(f'dvc pull {dvc_file}')
    df = pd.read_csv(df_path)

    # Ensure required metrics are present
    required_cols = ['HasClaim', 'ClaimFrequency', 'ClaimSeverity', 'Margin']
    if not all(col in df.columns for col in required_cols):
        from src.eda_analysis import compute_metrics
        df = compute_metrics(df)

    # Handling missing data
    df = df.dropna(subset=['TotalClaims', 'TotalPremium', 'CalculatedPremiumPerTerm'])
    df['PostalCode'] = df['PostalCode'].fillna(df['PostalCode'].mode()[0])

    # Feature engineering
    df['PolicyAge'] = (pd.to_datetime('2025-06-19') - pd.to_datetime(df['TransactionMonth'])).dt.days / 365
    df.drop(columns=['TransactionMonth'], inplace=True)  # <<== Add this line
    df = df.drop(columns=['PolicyID', 'CustomerName'], errors='ignore')  # Safely ignore if missing
    df['PremiumToClaimRatio'] = df['TotalPremium'] / df['TotalClaims'].replace(0, 1)
    
    # Encoding categorical variables
    categorical_cols = ['Province', 'Gender', 'VehicleType', 'make']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Claim severity model (only where claims exist)
    df_severity = df[df['TotalClaims'] > 0].copy()
    X_severity = df_severity.drop(['TotalClaims'], axis=1)
    y_severity = df_severity['TotalClaims']
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(
        X_severity, y_severity, test_size=0.2, random_state=42
    )

    # Claim probability model (binary target)
    df_prob = df.copy()
    y_prob = df_prob['HasClaim']
    X_prob = df_prob.drop(['HasClaim', 'TotalClaims'], axis=1)
    X_train_prob, X_test_prob, y_train_prob, y_test_prob = train_test_split(
        X_prob, y_prob, test_size=0.2, random_state=42
    )

    return X_train_sev, X_test_sev, y_train_sev, y_test_sev, X_train_prob, X_test_prob, y_train_prob, y_test_prob

def build_models(X_train_sev, y_train_sev, X_train_prob, y_train_prob) -> tuple:
    """Train regression and classification models."""
    models_sev = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }
    for model in models_sev.values():
        model.fit(X_train_sev, y_train_sev)

    models_prob = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
    }
    for model in models_prob.values():
        model.fit(X_train_prob, y_train_prob)

    return models_sev, models_prob

def evaluate_models(models_sev, models_prob, X_test_sev, y_test_sev, X_test_prob, y_test_prob) -> tuple:
    """Evaluate severity and probability models with metrics."""
    results_sev = {}
    for name, model in models_sev.items():
        y_pred = model.predict(X_test_sev)
        results_sev[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test_sev, y_pred)),
            'R²': r2_score(y_test_sev, y_pred)
        }

    results_prob = {}
    for name, model in models_prob.items():
        y_pred = model.predict(X_test_prob)
        results_prob[name] = {
            'Accuracy': accuracy_score(y_test_prob, y_pred),
            'Precision': precision_score(y_test_prob, y_pred),
            'Recall': recall_score(y_test_prob, y_pred),
            'F1': f1_score(y_test_prob, y_pred)
        }

    return results_sev, results_prob

def interpret_model(best_model, X_train, X_test, model_type: str, output_dir: str = '../plots') -> None:
    """Use SHAP to interpret model features and save a summary plot."""
    os.makedirs(output_dir, exist_ok=True)
    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, max_display=10, show=False)
    plt.title(f"SHAP Feature Importance for {model_type.capitalize()} Prediction")
    plt.savefig(os.path.join(output_dir, f'shap_{model_type}.png'))
    plt.close()

    feature_importance = dict(zip(X_test.columns, np.abs(shap_values.values).mean(axis=0)))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n=== Top 5 Influential Features for {model_type.capitalize()} ===")
    for feature, importance in top_features:
        print(f"{feature}: Importance = {importance:.4f}")
        if 'PolicyAge' in feature:
            print("  SHAP analysis: Older policy age increases predicted claims — consider age-based premium adjustments.")
        elif 'TotalPremium' in feature:
            print("  SHAP analysis: Higher premiums are associated with higher predicted claims — verify premium fairness.")
