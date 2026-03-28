import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PLOTS_DIR = PROJECT_ROOT / "dashboard" / "plots"

def generate_forecasts():
    print("Generating scenario-based emissions forecast...")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # WHAT: Load baseline dataset
    # WHY: We need a starting point (Year 0) to project future trends based on current averages.
    df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
    
    # Baseline average CO2 Emission
    baseline_co2 = df['CO2 Emissions(g/km)'].mean()
    
    years = np.arange(0, 11)  # Year 0 to Year 10
    
    # Scenarios Definition
    # Scenario 1: Baseline (status quo) 
    # Scenario 2: Increase EV adoption (gradually replaces 50% of ICE footprint over 10 years, drastically cutting CO2)
    # Scenario 3: Reduce engine sizes & improve fuel efficiency (2% annual compounding improvement in overall ICE efficiency)
    
    # WHAT: Simulating scenarios mathematically.
    # WHY: Allows us to see the macroscopic impact of policy or consumer shifts over a decade.
    emissions_baseline = [baseline_co2] * 11
    
    emissions_ev_shift = []
    current_co2 = baseline_co2
    for yr in years:
        # Assuming a linear trend down to 50% of current emissions in 10 years due to EV mix
        emissions_ev_shift.append(baseline_co2 * (1 - (0.05 * yr)))
        
    emissions_efficiency = []
    current_eff_co2 = baseline_co2
    for yr in years:
        # Compounding 2.5% emission reduction per year
        emissions_efficiency.append(current_eff_co2)
        current_eff_co2 *= 0.975
        
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(years, emissions_baseline, label='Status Quo (No Change)', linestyle='--', color='gray')
    plt.plot(years, emissions_ev_shift, label='Aggressive EV Adoption (50% by Yr 10)', marker='o', color='green')
    plt.plot(years, emissions_efficiency, label='ICE Improvements (2.5% per annum)', marker='s', color='blue')
    
    plt.title('Simulated Fleet Average CO2 Emissions Over 10 Years')
    plt.xlabel('Years from Present')
    plt.ylabel('Average Fleet CO2 Emissions (g/km)')
    plt.xticks(years)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = PLOTS_DIR / 'forecast_scenarios.png'
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Forecast scenario plot saved to {plot_path}")

if __name__ == "__main__":
    generate_forecasts()
