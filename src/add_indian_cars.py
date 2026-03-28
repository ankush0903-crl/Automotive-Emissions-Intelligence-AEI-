import pandas as pd
from pathlib import Path

def add_indian_cars():
    raw_file = Path("c:/Users/Ankush/OneDrive/Desktop/kpi/data/CO2 Emissions_Canada.csv")
    
    # Check if we already added them
    df = pd.read_csv(raw_file)
    if 'MARUTI SUZUKI' in df['Make'].values or 'MAHINDRA' in df['Make'].values:
        print("Indian cars already added!")
        return

    # Create synthetic data
    indian_cars = [
        ['MARUTI SUZUKI', 'SWIFT', 'COMPACT', 1.2, 4, 'M5', 'X', 6.0, 4.5, 5.3, 53, 120],
        ['MARUTI SUZUKI', 'BALENO', 'COMPACT', 1.2, 4, 'M5', 'X', 5.8, 4.3, 5.1, 55, 115],
        ['MARUTI SUZUKI', 'DZIRE', 'COMPACT', 1.2, 4, 'AS5', 'X', 5.9, 4.4, 5.2, 54, 118],
        ['MARUTI SUZUKI', 'BREZZA', 'SUV - SMALL', 1.5, 4, 'M5', 'X', 7.5, 5.5, 6.6, 43, 150],
        ['MARUTI SUZUKI', 'ERTIGA', 'MINIVAN', 1.5, 4, 'AS6', 'X', 7.8, 5.8, 6.9, 41, 158],
        ['MARUTI SUZUKI', 'CIAZ', 'MID-SIZE', 1.5, 4, 'M5', 'X', 7.0, 5.0, 6.1, 46, 140],
        ['MAHINDRA', 'THAR', 'SUV - SMALL', 2.2, 4, 'A6', 'D', 11.0, 8.5, 9.8, 29, 260],
        ['MAHINDRA', 'XUV700', 'SUV - STANDARD', 2.2, 4, 'A6', 'D', 10.5, 7.8, 9.2, 31, 240],
        ['MAHINDRA', 'SCORPIO-N', 'SUV - STANDARD', 2.2, 4, 'M6', 'D', 11.5, 8.8, 10.3, 27, 270],
        ['MAHINDRA', 'BOLERO', 'SUV - STANDARD', 1.5, 3, 'M5', 'D', 9.5, 7.5, 8.6, 33, 225],
        ['MAHINDRA', 'XUV300', 'SUV - SMALL', 1.2, 3, 'M6', 'X', 8.5, 6.0, 7.4, 38, 175]
    ]
    
    columns = [
        'Make', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders', 'Transmission', 'Fuel Type', 
        'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 
        'Fuel Consumption Comb (mpg)', 'CO2 Emissions(g/km)'
    ]
    
    new_df = pd.DataFrame(indian_cars, columns=columns)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(raw_file, index=False)
    print("Added Indian cars to dataset.")

if __name__ == "__main__":
    add_indian_cars()
