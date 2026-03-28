# Vehicle CO2 Emission Analysis, KPI Modeling, and Forecasting System 🌱

Welcome to the end-to-end data science project analyzing vehicle CO2 emissions! This project was built to be both **beginner-friendly and industry-level**. It breaks down the entire process from raw data preprocessing, exploratory data analysis (EDA), custom KPI mathematical modeling, and machine learning predictions, terminating in a beautiful **Streamlit interactive dashboard**.

---

## 📂 Project Structure

```text
kpi/
│
├── data/                    # The raw and processed datasets
│   ├── CO2 Emissions_Canada.csv
│   └── processed/           # Created dynamically by data_processing.py
│
├── src/                     # Core Python Pipeline Scripts
│   ├── data_processing.py   # Cleans data, removes outliers, scales & encodes features
│   ├── kpi_creation.py      # Calculates and normalizes custom VEEI formula
│   ├── eda.py               # Generates plots and extracts mathematical insights
│   ├── train_models.py      # Trains LR, RF, XGBoost and selects best performing model
│   └── forecast.py          # Simulates 10-year CO2 trend scenarios
│
├── models/                  # Stored artifacts
│   ├── best_co2_model.joblib
│   └── model_features.joblib
│
├── dashboard/               # Streamlit Application
│   ├── app.py               # Interactive UI
│   ├── plots/               # Pre-rendered EDA and forecast images
│   └── eda_insights.md      # Textual insights from EDA
│
├── requirements.txt         # Project dependencies
└── README.md                # You are here!
```

---

## 🚀 How to Run the Project

1. **Activate the Virtual Environment:**
   If you aren't already in the virtual environment `venv`, activate it:
   - On Windows: `.\venv\Scripts\activate`
   - On Mac/Linux: `source venv/bin/activate`

2. **Run the Streamlit Dashboard:**
   The entire application culminates in the interactive dashboard. Run:
   ```bash
   streamlit run dashboard/app.py
   ```
   *Streamlit will automatically open a tab in your browser (usually `http://localhost:8501`).*

*(Optional)* You can manually re-run the pipeline scripts in this order:
`python src/data_processing.py`
`python src/kpi_creation.py`
`python src/eda.py`
`python src/train_models.py`
`python src/forecast.py`

---

## 🧩 Explaining the Steps (What & Why)

### 1. Data Preprocessing (`src/data_processing.py`)
- **What:** Loads the CO2 emissions dataset, removes missing values and statistical outliers (via IQR ranges), scales continuous features (using `StandardScaler`), and encodes categorical variables (`OneHotEncoding`).
- **Why:** Machine Learning algorithms require numbers (not text like "Sedan") and perform drastically better when extreme, unrepresentative outliers (e.g., million-dollar hypercars) are removed. Scaling ensures variables like "Cylinders" (scale 4-12) don't overpower "Engine Size" (scale 1.5-6.0) simply because of raw numeric sizing.

### 2. Vehicle Emission Efficiency Index (VEEI) (`src/kpi_creation.py`)
- **What:** Formula: `(Fuel Cons Comb * Engine Size * Cylinders) / (Comb MPG * Fuel Type Factor)`
  The result is then inverted and normalized on a scale from 0 to 100.
- **Why:** CO2 Emissions measured in raw `g/km` lacking context. VEEI measures the **relative efficiency** of a car given its mechanical size. A higher score means it performs optimally for its class, while a lower score means it is a highly inefficient "gas guzzler." 

### 3. Exploratory Data Analysis (`src/eda.py`)
- **What:** Plots correlation matrices, histograms, and trains a quick Random Forest to ascertain feature importances.
- **Why:** Understanding our data guides modeling. We discovered that **Engine Size** and **Fuel Consumption** have an unsurprising, massive positive correlation with emissions.

### 4. Machine Learning (`src/train_models.py`)
- **What:** Trained three models: `Linear Regression`, `Random Forest`, and `XGBoost`. Evaluated using $R^2$, RMSE, and MAE. 
- **Why:** By comparing linear (Linear Regression) against tree-based ensemble models (Random Forest, XGBoost), we map non-linear behaviors between vehicle characteristics. `XGBoost` proved best ($R^2$: `0.998`) and was saved via `joblib`.

### 5. Forecasting (`src/forecast.py`)
- **What:** Simulated future baselines looking 10-years ahead using arbitrary constraints (Aggressive EV adoption vs. Incremental ICE efficiency improvements). 
- **Why:** Predictive analytics is not just about a single vehicle, but macroeconomic trends. This allows policy-makers or enterprise users to scenario-plan global fleet emissions. 

---
### Happy Predicting! 🚗💨
