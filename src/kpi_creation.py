import pandas as pd
from pathlib import Path

# WHAT: Define setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

def create_kpi(file_path):
    print(f"Loading cleaned data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # WHAT: Calculate Vehicle Emission Efficiency Index (VEEI)
    # Formula: VEEI = (Fuel Consumption Combined * Engine Size * Cylinders) / (Fuel Efficiency Score * Fuel Type Factor)
    # WHY: This custom KPI balances a vehicle's raw size/power (Engine Size, Cylinders) against its efficiency (mpg) 
    # and fuel dirtiness (Fuel Type Factor). A lower VEEI means better emission efficiency relative to its size.
    # Note: Fuel Efficiency Score = Fuel Consumption Comb (mpg)
    
    # To prevent division by zero, we ensure denominator is > 0
    denominator = df['Fuel Consumption Comb (mpg)'] * df['Fuel Type Factor']
    denominator = denominator.replace(0, 1) # Fallback just in case
    
    numerator = df['Fuel Consumption Comb (L/100 km)'] * df['Engine Size(L)'] * df['Cylinders']
    
    df['VEEI_raw'] = numerator / denominator
    
    # WHAT: Normalize the VEEI to a 0-100 scale.
    # WHY: A 0-100 scale is much easier for users to understand in a dashboard. 
    # We will invert it so 100 = Best Efficiency, 0 = Worst Efficiency.
    min_veei = df['VEEI_raw'].min()
    max_veei = df['VEEI_raw'].max()
    
    # Normalizing to 0-100 where higher is better:
    # First, scale to 0-100
    df['VEEI'] = 100 - (((df['VEEI_raw'] - min_veei) / (max_veei - min_veei)) * 100)
    
    df.drop(columns=['VEEI_raw'], inplace=True)
    
    output_path = PROCESSED_DATA_DIR / "data_with_kpi.csv"
    df.to_csv(output_path, index=False)
    print(f"Data with KPI saved to {output_path}")
    
    # Let's print the top 5 BEST vehicles according to our KPI
    print("\nTop 5 Most Emission-Efficient Vehicles (Highest VEEI):")
    top_5 = df.sort_values(by='VEEI', ascending=False).head(5)
    print(top_5[['Make', 'Model', 'VEEI', 'CO2 Emissions(g/km)']])

if __name__ == "__main__":
    cleaned_file = PROCESSED_DATA_DIR / "cleaned_data.csv"
    create_kpi(cleaned_file)
