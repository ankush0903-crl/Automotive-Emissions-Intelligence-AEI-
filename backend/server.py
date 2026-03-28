import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

try:
    clean_df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
    model = joblib.load(MODELS_DIR / 'best_co2_model.joblib')
    model_features = joblib.load(MODELS_DIR / 'model_features.joblib')
    print("Models and data loaded successfully.")
except Exception as e:
    print(f"Error loading models/data: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/options', methods=['GET'])
def get_options():
    try:
        makes = sorted(clean_df['Make'].dropna().unique().tolist())
        v_classes = sorted(clean_df['Vehicle Class'].dropna().unique().tolist())
        transmissions = sorted(clean_df['Transmission'].dropna().unique().tolist())
        fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']
        
        return jsonify({
            'makes': makes,
            'v_classes': v_classes,
            'transmissions': transmissions,
            'fuel_types': fuel_types
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        make = data.get('make')
        v_class = data.get('vehicle_class')
        transmission = data.get('transmission')
        fuel_type = data.get('fuel_type')
        engine_size = float(data.get('engine_size', 2.0))
        cylinders = int(data.get('cylinders', 4))
        fuel_city = float(data.get('fuel_city', 10.0))
        fuel_hwy = float(data.get('fuel_hwy', 7.0))
        fuel_comb = float(data.get('fuel_comb', 8.5))
        fuel_mpg = int(data.get('fuel_mpg', 30))
        
        fuel_factor_map = {'Petrol': 1.0, 'Diesel': 0.9, 'Electric': 0.8, 'Hybrid': 0.8}
        fuel_factor = fuel_factor_map.get(fuel_type, 1.0)
        aerodynamic_proxy = fuel_hwy
        city_hwy_diff = fuel_city - fuel_hwy
        
        user_data = {
            'Make': [make],
            'Vehicle Class': [v_class],
            'Transmission': [transmission],
            'Fuel Type': [fuel_type],
            'Engine Size(L)': [engine_size],
            'Cylinders': [cylinders],
            'Fuel Consumption City (L/100 km)': [fuel_city],
            'Fuel Consumption Hwy (L/100 km)': [fuel_hwy],
            'Fuel Consumption Comb (L/100 km)': [fuel_comb],
            'Fuel Consumption Comb (mpg)': [fuel_mpg],
            'Fuel Type Factor': [fuel_factor],
            'Aerodynamic Proxy': [aerodynamic_proxy],
            'City_Hwy_Diff': [city_hwy_diff]
        }
        
        user_df = pd.DataFrame(user_data)
        
        combined = pd.concat([clean_df.drop(columns=['CO2 Emissions(g/km)'], errors='ignore'), user_df], ignore_index=True)
        cat_cols = ['Make', 'Vehicle Class', 'Transmission', 'Fuel Type']
        combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
        
        num_cols = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
                    'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
                    'Fuel Consumption Comb (mpg)', 'Fuel Type Factor', 'Aerodynamic Proxy', 'City_Hwy_Diff']
        
        scaler = StandardScaler()
        combined_encoded[num_cols] = scaler.fit_transform(combined_encoded[num_cols])
        
        user_processed = combined_encoded.iloc[[-1]].copy()
        
        for col in model_features:
            if col not in user_processed.columns:
                user_processed[col] = 0
                
        user_processed = user_processed[model_features]
        
        prediction = float(model.predict(user_processed)[0])
        
        clean_kpi = clean_df.copy()
        num = clean_kpi['Fuel Consumption Comb (L/100 km)'] * clean_kpi['Engine Size(L)'] * clean_kpi['Cylinders']
        den = clean_kpi['Fuel Consumption Comb (mpg)'] * clean_kpi['Fuel Type Factor'].replace(0, 1)
        kpi_raw_series = num / den
        min_veei = float(kpi_raw_series.min())
        max_veei = float(kpi_raw_series.max())
        
        user_num = fuel_comb * engine_size * cylinders
        user_den = (fuel_mpg * fuel_factor) if (fuel_mpg * fuel_factor) > 0 else 1
        user_veei_raw = user_num / user_den
        user_veei = 100 - (((user_veei_raw - min_veei) / (max_veei - min_veei)) * 100)
        user_veei = max(0.0, min(100.0, user_veei))
        
        if user_veei >= 80:
            interp = "Optimal Efficiency: This vehicle configuration achieves an exceptional balance of dimensional performance and emission control."
            status = "excellent"
        elif user_veei >= 50:
            interp = "Standard Efficiency: This vehicle performs within expected industry baselines. Tactical improvements could involve optimizing highway dynamics or reducing raw consumption metrics."
            status = "average"
        else:
            interp = "Suboptimal Performance: This configuration emits at volumes disproportionate to its operational specifications. Transitioning to alternative drivetrains represents a viable optimization pathway."
            status = "poor"
            
        return jsonify({
            'prediction': prediction,
            'veei': user_veei,
            'interpretation': interp,
            'status': status
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
