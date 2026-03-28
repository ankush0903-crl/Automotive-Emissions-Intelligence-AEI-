import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae

def train_and_evaluate():
    print("Loading ML-ready data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "ml_ready_data.csv")
    
    # Target and Features
    target = 'CO2 Emissions(g/km)'
    # Drop highly collinear fuel consumption metrics to avoid multicollinearity 
    # and keep 'Fuel Consumption Comb (L/100 km)' as the main proxy.
    X = df.drop(columns=[target, 'Model', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)'], errors='ignore')
    y = df[target]
    
    # 1. Prepare data for modeling
    # WHAT: Train/test split (80/20)
    # WHY: Need unseen data to evaluate how well the model generalizes.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost Regressor": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    best_name = ""
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 2 & 3. Train models and Evaluate
    # WHAT: Train three different algorithms and calculate R2, RMSE, MAE.
    # WHY: Comparing linear vs non-linear algorithms based on standardized metrics.
    with open(MODELS_DIR / "model_evaluation_report.md", "w") as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("## Metrics Explained\n")
        f.write("- **R2 Score**: Represents the proportion of variance in the dependent variable explained by the independent variables. Higher is better (max 1.0).\n")
        f.write("- **RMSE**: Root Mean Squared Error. Represents the average prediction error in the same units as the target variable. Lower is better, heavily penalizes large errors.\n")
        f.write("- **MAE**: Mean Absolute Error. The average absolute difference between prediction and actual. Lower is better.\n\n")
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            r2, rmse, mae = evaluate_model(y_test, y_pred)
            results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
            
            f.write(f"### {name}\n")
            f.write(f"- **R2 Score**: {r2:.4f}\n")
            f.write(f"- **RMSE**: {rmse:.4f}\n")
            f.write(f"- **MAE**: {mae:.4f}\n\n")
            
            print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name

        f.write(f"## Conclusion\n")
        f.write(f"The best performing model is **{best_name}** with an R2 score of {best_r2:.4f}. ")
        f.write(f"This model was chosen because it achieved the highest explanatory power (R2) while minimizing average errors (RMSE/MAE). ")
        f.write(f"Tree-based models (like Random Forest and XGBoost) typically capture non-linear interactions better than simple Linear Regression.\n")

    # Save the best model
    # WHAT: Save trained model using joblib.
    # WHY: So the Streamlit dashboard can load and use it for real-time predictions without retraining.
    model_path = MODELS_DIR / "best_co2_model.joblib"
    joblib.dump(best_model, model_path)
    
    # Save the feature columns so the dashboard knows the expected inputs exactly
    joblib.dump(list(X.columns), MODELS_DIR / "model_features.joblib")
    
    print(f"\nSaved Best Model ({best_name}) to {model_path}.")
    print("Report written to models/model_evaluation_report.md")

if __name__ == "__main__":
    train_and_evaluate()
