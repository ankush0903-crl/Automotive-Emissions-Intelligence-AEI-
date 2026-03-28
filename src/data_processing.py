import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Explain what + why
# WHAT: Define project paths.
# WHY: Ensures the script can find data regardless of where it's executed from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def process_data(file_path):
    print(f"Loading data from {file_path}...")
    # WHAT: Load dataset.
    # WHY: Initial step to bring data into memory for manipulation.
    df = pd.read_csv(file_path)
    
    # 1. Handle missing values
    # WHAT: Drop rows with missing values.
    # WHY: Missing values can cause errors during model training. In this dataset, we expect minimal missing values.
    initial_shape = df.shape
    df.dropna(inplace=True)
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows with missing values.")
    
    # WHAT: Filter out invalid transmission types
    invalid_transmissions = ["AV10", "AV6", "AV7", "AV8", "M6", "M7", "M5", "AM8", "AM9", "AM7"]
    pre_filter_shape = df.shape
    df = df[~df['Transmission'].isin(invalid_transmissions)]
    print(f"Dropped {pre_filter_shape[0] - df.shape[0]} rows with invalid transmissions.")
    
    # 2. Derived Features
    # WHAT: Map raw fuel types to common names (Petrol, Diesel, Electric, Hybrid)
    fuel_name_map = {'Z': 'Petrol', 'X': 'Petrol', 'D': 'Diesel', 'E': 'Electric', 'N': 'Hybrid'}
    df['Fuel Type'] = df['Fuel Type'].map(fuel_name_map).fillna('Petrol')

    # WHAT: Create 'Fuel Type Factor' based on mapped Fuel Type column.
    # WHY: Models need numerical representations of how "dirty" a fuel type is intrinsically.
    fuel_factor_map = {'Petrol': 1.0, 'Diesel': 0.9, 'Electric': 0.8, 'Hybrid': 0.8}
    df['Fuel Type Factor'] = df['Fuel Type'].map(fuel_factor_map).fillna(1.0)
    
    # WHAT: Aerodynamic proxy using Highway fuel consumption
    # WHY: Highway driving efficiency is heavily influenced by aerodynamics.
    df['Aerodynamic Proxy'] = df['Fuel Consumption Hwy (L/100 km)']
    
    # WHAT: City vs Highway difference
    # WHY: A large difference indicates a vehicle that is much less efficient in stop-and-go traffic (e.g. heavy vehicles).
    df['City_Hwy_Diff'] = df['Fuel Consumption City (L/100 km)'] - df['Fuel Consumption Hwy (L/100 km)']
    
    # 3. Remove Outliers (IQR method)
    # WHAT: Remove extreme outliers in CO2 emissions.
    # WHY: Outliers (like highly specialized supercars) skew ML models and averages, reducing generalization to normal cars.
    Q1 = df['CO2 Emissions(g/km)'].quantile(0.25)
    Q3 = df['CO2 Emissions(g/km)'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (df['CO2 Emissions(g/km)'] >= lower_bound) & (df['CO2 Emissions(g/km)'] <= upper_bound)
    df = df[outlier_mask].copy()
    print(f"Kept {df.shape[0]} rows after removing outliers.")
    
    # WHAT: One-hot encode categorical features ('Make', 'Vehicle Class', 'Transmission', 'Fuel Type')
    # WHY: ML algorithms require numerical inputs. One-hot encoding avoids assigning false ordinal relationships between categories.
    cat_cols = ['Make', 'Vehicle Class', 'Transmission', 'Fuel Type']
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # WHAT: Scale numerical features
    # WHY: Models like Linear Regression are sensitive to feature scales. Scaling ensures features contribute equally.
    num_cols = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
                'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
                'Fuel Consumption Comb (mpg)', 'Fuel Type Factor', 'Aerodynamic Proxy', 'City_Hwy_Diff']
    scaler = StandardScaler()
    df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    
    # We will keep the original columns for EDA, but create an '_encoded' dataframe for ML.
    clean_data_path = DATA_DIR / "processed" / "cleaned_data.csv"
    encoded_data_path = DATA_DIR / "processed" / "ml_ready_data.csv"
    
    os.makedirs(clean_data_path.parent, exist_ok=True)
    
    df.to_csv(clean_data_path, index=False)
    df_encoded.to_csv(encoded_data_path, index=False)
    
    print(f"Cleaned data saved to {clean_data_path}")
    print(f"ML Ready encoded data saved to {encoded_data_path}")
    
    # Return so KPI script can use the cleaned data
    return clean_data_path

if __name__ == "__main__":
    raw_file = DATA_DIR / "CO2 Emissions_Canada.csv"
    process_data(raw_file)
