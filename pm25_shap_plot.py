import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os 

# --- Parameters Configuration ---
N_HOURS_AHEAD = 24
# Load the optimized model
MODEL_FILENAME = 'pm25_xgb_optimized_predict_24hr_2019_2023_train.joblib' 
FILE_PATH = "la_pm25_combined_clean_v15_2019_2024.csv" 
SPLIT_DATE = pd.to_datetime('2024-01-01')

# Define feature columns
FEATURE_AND_TARGET_COLUMNS = [
    'PM25', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 
    'HourlyWindSpeed', 'HourlyDewPointTemperature', 'HourlyStationPressure', 
    'HourlyVisibility' 
]

# Set Chinese font to avoid garbled characters (for local plotting)
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
except:
    print("🚨 Warning: Font configuration failed. Chart labels may still be garbled.")

# --- Data Preparation (Must match the training script exactly) ---

try:
    df = pd.read_csv(FILE_PATH, index_col='DateTime', parse_dates=True)
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    for col in FEATURE_AND_TARGET_COLUMNS:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    original_len = len(df)
    FEATURE_AND_TARGET_COLUMNS = [c for c in FEATURE_AND_TARGET_COLUMNS if c in df.columns and df[c].isnull().sum() != original_len]
    df = df[FEATURE_AND_TARGET_COLUMNS].copy()
    df = df.fillna(method='ffill').fillna(method='bfill').dropna() 
    
    target_variable = 'PM25'
    y = df[target_variable].shift(-N_HOURS_AHEAD)
    X = df[[c for c in FEATURE_AND_TARGET_COLUMNS if c != 'PM25']].copy()
    
    # Feature Engineering
    X[f'PM25_Lag_{N_HOURS_AHEAD}'] = df[target_variable].shift(N_HOURS_AHEAD)
    X[f'PM25_RollingMean_24hr_Lag_{N_HOURS_AHEAD}'] = df[target_variable].rolling(window=24).mean().shift(N_HOURS_AHEAD)
    X['DayOfWeek'] = X.index.dayofweek
    X['Month'] = X.index.month
    X['Hour'] = X.index.hour
    X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
    X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
    X = X.drop('Hour', axis=1) 
    
    combined_df = pd.concat([X, y.rename('Target_PM25')], axis=1).dropna() 
    X = combined_df.drop('Target_PM25', axis=1)
    
    # Split the Test Set
    X_test = X[X.index >= SPLIT_DATE]
    # Clean column names for XGBoost
    X_test.columns = [c.replace(':', '_').replace('<', '_') for c in X_test.columns]

except Exception as e:
    print(f"🚨 Error during data preparation: {e}")
    exit()

# --- Load Model and Calculate SHAP ---

try:
    if not os.path.exists(MODEL_FILENAME):
        print(f"🚨 Error: Model file {MODEL_FILENAME} not found. Ensure the training script has run successfully.")
        exit()
        
    model = joblib.load(MODEL_FILENAME)
    print(f"✅ Successfully loaded optimized model: {MODEL_FILENAME}")
    
    # Sample a subset (500 samples) to speed up SHAP calculation
    X_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    
    # Initialize SHAP
    print("⚙️ Starting SHAP value calculation... (This may take a few minutes)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # Plot Single Feature Dependence: HourlyDewPointTemperature
    plt.figure(figsize=(8, 6))
    shap.dependence_plot("HourlyDewPointTemperature", shap_values, X_test_sample, 
                         interaction_index=None, show=False)
    plt.title('Impact of Dew Point Temperature on PM2.5 Prediction (SHAP Dependence)')
    
    # Save the plot
    filename = 'shap_dependence_dewpoint_optimized.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"\n✅ SHAP dependence plot generated and saved as {filename}!")
    
except Exception as e:
    print(f"🚨 SHAP plotting failed. Error: {e}")
