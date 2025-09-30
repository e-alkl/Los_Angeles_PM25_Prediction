import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os # 新增：用於確認文件路徑

# --- 參數設定 ---
N_HOURS_AHEAD = 24
# 載入優化模型
MODEL_FILENAME = 'pm25_xgb_optimized_predict_24hr_2019_2023_train.joblib' 
FILE_PATH = "la_pm25_combined_clean_v15_2019_2024.csv" 
SPLIT_DATE = pd.to_datetime('2024-01-01')

# 定義特徵欄位
FEATURE_AND_TARGET_COLUMNS = [
    'PM25', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 
    'HourlyWindSpeed', 'HourlyDewPointTemperature', 'HourlyStationPressure', 
    'HourlyVisibility' 
]

# 亂碼修正
try:
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'Arial Unicode MS']
    mpl.rcParams['axes.unicode_minus'] = False
except:
    print("🚨 警告：字體設定失敗，圖表標籤可能仍有亂碼。")

# --- 數據準備 (確保與訓練時完全一致) ---

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
    
    # 特徵工程
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
    
    # 劃分測試集
    X_test = X[X.index >= SPLIT_DATE]
    X_test.columns = [c.replace(':', '_').replace('<', '_') for c in X_test.columns]

except Exception as e:
    print(f"🚨 數據準備階段發生錯誤: {e}")
    exit()

# --- 載入模型與 SHAP 計算 ---

try:
    if not os.path.exists(MODEL_FILENAME):
        print(f"🚨 錯誤：找不到模型文件 {MODEL_FILENAME}。請確保訓練腳本已成功運行並儲存了優化模型。")
        exit()
        
    model = joblib.load(MODEL_FILENAME)
    print(f"✅ 成功載入優化模型：{MODEL_FILENAME}")
    
    # 抽取子集 (500 樣本) 以加速計算
    X_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    
    # 初始化 SHAP
    print("⚙️ 開始計算 SHAP 值... (這可能需要幾分鐘)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_sample)

    # 繪製單一特徵依賴圖：HourlyDewPointTemperature
    plt.figure(figsize=(8, 6))
    shap.dependence_plot("HourlyDewPointTemperature", shap_values, X_test_sample, 
                         interaction_index=None, show=False)
    plt.title('露點溫度對 PM2.5 預測的影響 (SHAP 依賴圖)')
    
    # 儲存圖片
    filename = 'shap_dependence_dewpoint_optimized.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"\n✅ SHAP 依賴圖已生成並儲存為 {filename}！")
    
except Exception as e:
    print(f"🚨 SHAP 繪製失敗。錯誤: {e}")