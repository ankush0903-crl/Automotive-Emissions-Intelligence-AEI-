import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler

# WHAT: Define essential paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "dashboard" / "plots"

st.set_page_config(page_title="Vehicle CO2 Emissions & KPI Dashboard", layout="wide")

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Hide Streamlit default UI components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Typography and layout */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .main-title {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }

    /* Metric Cards */
    .metric-container {
        display: flex;
        gap: 1.5rem;
        margin: 2rem 0;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 1.5rem;
        flex: 1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
    }
    .metric-unit {
        font-size: 1.2rem;
        color: #64748b;
        font-weight: 500;
    }
    
    .text-co2 { color: #f87171 !important; }
    .text-veei { color: #34d399 !important; }

    /* Interpretation Callouts */
    .interp {
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin-top: 1rem;
        background: rgba(255,255,255,0.02);
        color: #e2e8f0;
        font-size: 1.05rem;
    }
    .interp.excellent { border-left-color: #34d399; }
    .interp.average { border-left-color: #fbbf24; background: rgba(251, 191, 36, 0.05); }
    .interp.poor { border-left-color: #f87171; background: rgba(248, 113, 113, 0.05); }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    clean_df = pd.read_csv(PROCESSED_DATA_DIR / "cleaned_data.csv")
    kpi_df = pd.read_csv(PROCESSED_DATA_DIR / "data_with_kpi.csv")
    return clean_df, kpi_df

@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "best_co2_model.joblib")
    features = joblib.load(MODELS_DIR / "model_features.joblib")
    return model, features

clean_df, kpi_df = load_data()
model, model_features = load_model()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Vehicle Parameters")
# WHAT: Provide user controls for all required features
# WHY: Allows users to interactively test how different specs change CO2 and the KPI.

make = st.sidebar.selectbox("Make", clean_df['Make'].unique())
v_class = st.sidebar.selectbox("Vehicle Class", clean_df['Vehicle Class'].unique())
transmission = st.sidebar.selectbox("Transmission", clean_df['Transmission'].unique())
fuel_type = st.sidebar.selectbox("Fuel Type", clean_df['Fuel Type'].unique())

is_ev_or_hybrid = fuel_type in ['Electric', 'Hybrid']

engine_size = st.sidebar.number_input("Engine Size (L)", min_value=0.0 if is_ev_or_hybrid else 0.5, max_value=10.0, value=0.0 if is_ev_or_hybrid else 2.0)
cylinders = st.sidebar.number_input("Cylinders", min_value=0 if is_ev_or_hybrid else 2, max_value=16, value=0 if is_ev_or_hybrid else 4)
fuel_city = st.sidebar.number_input("Fuel Consumption City (L/100 km)", min_value=0.0 if is_ev_or_hybrid else 1.0, value=0.0 if is_ev_or_hybrid else 10.0)
fuel_hwy = st.sidebar.number_input("Fuel Consumption Hwy (L/100 km)", min_value=0.0 if is_ev_or_hybrid else 1.0, value=0.0 if is_ev_or_hybrid else 7.0)
fuel_comb = st.sidebar.number_input("Fuel Consumption Comb (L/100 km)", min_value=0.0 if is_ev_or_hybrid else 1.0, value=0.0 if is_ev_or_hybrid else 8.5)
fuel_mpg = st.sidebar.number_input("Fuel Consumption Comb (mpg)", min_value=0 if is_ev_or_hybrid else 10, value=0 if is_ev_or_hybrid else 30)

# Derived features for prediction
fuel_factor_map = {'Petrol': 1.0, 'Diesel': 0.9, 'Electric': 0.8, 'Hybrid': 0.8}
fuel_factor = fuel_factor_map.get(fuel_type, 1.0)
aerodynamic_proxy = fuel_hwy
city_hwy_diff = fuel_city - fuel_hwy

# --- MAIN APP ---
st.markdown("<h1 class='main-title'>Vehicle CO2 Emission & KPI Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced predictive engine for automotive carbon footprint modeling and standardizing efficiency through the Vehicle Emission Efficiency Index (VEEI).</p>", unsafe_allow_html=True)

# WHAT: Create a single user-input dictionary for processing.
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

st.subheader("1. Real-Time CO2 Prediction & KPI")

if st.button("Predict CO2 Emission & Calculate VEEI"):
    # --- ML PREDICTION PREP ---
    # Combine with original data briefly to ensure all dummy columns are created correctly
    combined = pd.concat([clean_df.drop(columns=['CO2 Emissions(g/km)'], errors='ignore'), user_df], ignore_index=True)
    cat_cols = ['Make', 'Vehicle Class', 'Transmission', 'Fuel Type']
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)
    
    # WHAT: Refitting scaler on the fly for UI simplicity
    num_cols = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)',
                'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
                'Fuel Consumption Comb (mpg)', 'Fuel Type Factor', 'Aerodynamic Proxy', 'City_Hwy_Diff']
    
    scaler = StandardScaler()
    combined_encoded[num_cols] = scaler.fit_transform(combined_encoded[num_cols])
    
    # Get the processed user row
    user_processed = combined_encoded.iloc[[-1]]
    
    # Ensure all training features exist
    for col in model_features:
        if col not in user_processed.columns:
            user_processed[col] = 0
            
    # Keep only model features in exact order
    user_processed = user_processed[model_features]
    
    # PREDICT
    prediction = model.predict(user_processed)[0]
    
    # --- KPI CALCULATION ---
    # To normalize correctly, compute raw VEEI for the entire clean dataset
    clean_kpi = clean_df.copy()
    num = clean_kpi['Fuel Consumption Comb (L/100 km)'] * clean_kpi['Engine Size(L)'] * clean_kpi['Cylinders']
    den = clean_kpi['Fuel Consumption Comb (mpg)'] * clean_kpi['Fuel Type Factor'].replace(0, 1)
    kpi_raw_series = num / den
    min_veei = kpi_raw_series.min()
    max_veei = kpi_raw_series.max()
    
    # User KPI
    user_num = fuel_comb * engine_size * cylinders
    user_den = (fuel_mpg * fuel_factor) if (fuel_mpg * fuel_factor) > 0 else 1
    user_veei_raw = user_num / user_den
    user_veei = 100 - (((user_veei_raw - min_veei) / (max_veei - min_veei)) * 100)
    
    # Cap between 0 and 100
    user_veei = max(0, min(100, user_veei))
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-title">Predicted CO2 Emissions</div>
            <div class="metric-value text-co2">{{prediction:.1f}} <span class="metric-unit">g/km</span></div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Efficiency Index (VEEI)</div>
            <div class="metric-value text-veei">{{user_veei:.1f}} <span class="metric-unit">/ 100</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Analytical Interpretation")
    if user_veei >= 80:
        st.markdown("<div class='interp excellent'><strong>Optimal Efficiency:</strong> This vehicle configuration achieves an exceptional balance of dimensional performance and emission control.</div>", unsafe_allow_html=True)
    elif user_veei >= 50:
        st.markdown("<div class='interp average'><strong>Standard Efficiency:</strong> This vehicle performs within expected industry baselines. Tactical improvements could involve optimizing highway dynamics or reducing raw consumption metrics.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='interp poor'><strong>Suboptimal Performance:</strong> This configuration emits at volumes disproportionate to its operational specifications. Transitioning to alternative drivetrains represents a viable optimization pathway.</div>", unsafe_allow_html=True)
        
st.divider()

st.subheader("2. Exploratory Data Analysis (EDA)")
st.markdown("Insights derived from analyzing vehicle specifications versus CO2 emissions. The correlation matrix shows that Engine Size and Fuel Consumption are the primary drivers of high emissions.")

col3, col4 = st.columns(2)
# Load EDA plots safely
try:
    with col3:
        st.image(Image.open(PLOTS_DIR / 'correlation_heatmap.png'), caption="Correlation Heatmap")
        st.image(Image.open(PLOTS_DIR / 'co2_distribution.png'), caption="CO2 Emissions Distribution")
    with col4:
        st.image(Image.open(PLOTS_DIR / 'feature_importance.png'), caption="Random Forest Feature Importance")
        st.image(Image.open(PLOTS_DIR / 'engine_size_distribution.png'), caption="Engine Size Distribution")
except Exception as e:
    st.warning("EDA plots not found. Please ensure the EDA script ran successfully.")

st.divider()

st.subheader("3. Future Emission Scenarios (10-Year Forecast)")
st.markdown("Using baseline CO2 emission averages, we simulated three hypothetical future paths for the auto industry:")
try:
    st.image(Image.open(PLOTS_DIR / 'forecast_scenarios.png'), use_column_width=True, caption="10-Year Simulation")
except Exception as e:
    st.warning("Forecast plot not found.")
