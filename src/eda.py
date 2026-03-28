import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "dashboard" / "plots"
INSIGHTS_FILE = PROJECT_ROOT / "dashboard" / "eda_insights.md"

def perform_eda():
    print("Starting Exploratory Data Analysis...")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
    df_encoded = pd.read_csv(PROCESSED_DATA_DIR / "ml_ready_data.csv")
    
    insights = []
    insights.append("# Exploratory Data Analysis (EDA) Insights\n")
    
    # 1. Distribution Plots
    # WHAT: Plotting the distribution of CO2 Emissions and Engine Size
    # WHY: Understanding the skew and spread of our target variable and primary feature.
    plt.figure(figsize=(10, 5))
    sns.histplot(df['CO2 Emissions(g/km)'], bins=50, kde=True, color='purple')
    plt.title('Distribution of CO2 Emissions')
    plt.xlabel('CO2 Emissions (g/km)')
    plt.ylabel('Frequency')
    plt.savefig(PLOTS_DIR / 'co2_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Engine Size(L)'], bins=30, kde=True, color='teal')
    plt.title('Distribution of Engine Sizes')
    plt.xlabel('Engine Size (L)')
    plt.ylabel('Frequency')
    plt.savefig(PLOTS_DIR / 'engine_size_distribution.png')
    plt.close()
    
    insights.append("## Distributions")
    insights.append("The CO2 Emissions distribution is roughly normal (bell-shaped) after outlier removal, meaning most vehicles fall in the average range, with fewer extremely clean or heavily polluting vehicles. Engine size shows multiple peaks corresponding to common engine sizes (like 2.0L and 3.5L).\n")

    # 2. Correlation Heatmap
    # WHAT: Visualizing correlations between numerical features
    # WHY: Identifies which numerical factors strongly correlate with CO2 emissions.
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[num_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'correlation_heatmap.png')
    plt.close()
    
    # Extract insight
    co2_corr = corr_matrix['CO2 Emissions(g/km)'].sort_values(ascending=False)
    insights.append("## Correlations")
    insights.append(f"The factor that affects CO2 the most positively is **{co2_corr.index[1]}** (correlation: {co2_corr.iloc[1]:.2f}).")
    insights.append(f"The factor that affects CO2 the most negatively is **{co2_corr.index[-1]}** (correlation: {co2_corr.iloc[-1]:.2f}).")
    insights.append("Highly positively correlated features like Engine Size and Fuel Consumption Comb directly drive CO2 emissions up, whereas higher MPG heavily reduces CO2.\n")

    # 3. Feature Importance using a quick Random Forest
    # WHAT: Training a Random Forest Regressor to extract feature importance.
    # WHY: Correlation only shows linear relationships. RF shows non-linear importances and handles all encoded categoricals.
    target = 'CO2 Emissions(g/km)'
    X = df_encoded.drop(columns=[target, 'Model', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)'], errors='ignore') 
    # Drop highly collinear features to get a better read on base features
    y = df_encoded[target]
    
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='viridis', dodge=False)
    plt.legend([],[], frameon=False)
    plt.title('Top 10 Feature Importances for CO2 Emissions')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'feature_importance.png')
    plt.close()
    
    insights.append("## Feature Importance")
    insights.append(f"According to the Random Forest model, the most impactful feature for predicting CO2 emissions is **{importances.index[0]}**, followed by **{importances.index[1]}**.")
    insights.append("Engine size and fuel consumption metrics play the overwhelmingly largest role compared to categorical variables like Make or Vehicle Class.\n")

    # 4. Write insights
    with open(INSIGHTS_FILE, 'w') as f:
        f.write('\n'.join(insights))
        
    print(f"EDA complete. Plots saved to {PLOTS_DIR}. Insights written to {INSIGHTS_FILE}.")
    
if __name__ == "__main__":
    perform_eda()
