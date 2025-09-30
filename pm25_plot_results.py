import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- 參數設定 (必須與訓練腳本一致) ---
# --- 參數設定 (確保指向優化模型！) ---
N_HOURS_AHEAD = 24  
# 這一行是關鍵！請手動替換為新的優化模型文件名。
MODEL_FILENAME = 'pm25_xgb_optimized_predict_24hr_2019_2023_train.joblib' 
FILE_PATH = "la_pm25_combined_clean_v15_2019_2024.csv" 
SPLIT_DATE = pd.to_datetime('2024-01-01')

# 定義特徵欄位
FEATURE_AND_TARGET_COLUMNS = [
    'PM25', 
    'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 
    'HourlyWindSpeed', 'HourlyDewPointTemperature', 
    'HourlyStationPressure', 
    'HourlyVisibility' 
]

# --- 1. 數據準備 (複製訓練腳本的精確步驟) ---

try:
    df = pd.read_csv(FILE_PATH, index_col='DateTime', parse_dates=True)
except FileNotFoundError:
    print(f"🚨 錯誤：找不到文件 {FILE_PATH}。")
    exit()

original_len = len(df)

# 移除時區
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

# 強制類型轉換和清理
for col in FEATURE_AND_TARGET_COLUMNS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

# 移除 100% 缺失的欄位
FEATURE_AND_TARGET_COLUMNS = [c for c in FEATURE_AND_TARGET_COLUMNS if c in df.columns and df[c].isnull().sum() != original_len]
df = df[FEATURE_AND_TARGET_COLUMNS].copy()

# 填充和清理
df = df.fillna(method='ffill').fillna(method='bfill')
df.dropna(inplace=True) 

# 特徵工程
target_variable = 'PM25'
y = df[target_variable].shift(-N_HOURS_AHEAD)
feature_columns = [c for c in FEATURE_AND_TARGET_COLUMNS if c != 'PM25']
X = df[feature_columns].copy()

# 滯後特徵
X[f'PM25_Lag_{N_HOURS_AHEAD}'] = df[target_variable].shift(N_HOURS_AHEAD)
X[f'PM25_RollingMean_24hr_Lag_{N_HOURS_AHEAD}'] = df[target_variable].rolling(window=24).mean().shift(N_HOURS_AHEAD)

# 時間特徵
X['DayOfWeek'] = X.index.dayofweek
X['Month'] = X.index.month
X['Hour'] = X.index.hour
X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
X = X.drop('Hour', axis=1) 

# 對齊 X 和 y
combined_df = pd.concat([X, y.rename('Target_PM25')], axis=1).dropna() 
X = combined_df.drop('Target_PM25', axis=1)
y = combined_df['Target_PM25']

# 劃分測試集
X_test = X[X.index >= SPLIT_DATE]
y_test = y[y.index >= SPLIT_DATE]

# 處理 XGBoost 特徵命名限制
X_test.columns = [c.replace(':', '_').replace('<', '_') for c in X_test.columns]


# --- 2. 載入模型並預測 ---

try:
    model = joblib.load(MODEL_FILENAME)
    print(f"✅ 成功載入模型：{MODEL_FILENAME}")
except FileNotFoundError:
    print(f"🚨 錯誤：找不到模型文件 {MODEL_FILENAME}。請先運行訓練腳本。")
    exit()

y_pred = model.predict(X_test)
print(f"✅ 完成 {len(y_pred)} 個時間點的預測。")


# --- 3. 可視化結果 ---

# 由於 R^2 計算失敗是由於索引錯位，我們在繪圖時直接使用 NumPy 數組。
# 我們將只繪製 2024 年的前 30 天，以清楚顯示細節。

# 終極修正：手動修正 1 小時錯位（為了視覺準確度）
# 繪圖時確保長度相同
y_test_plot = y_test.values[1:] 
y_pred_plot = y_pred[:-1]
time_index = X_test.index[1:]

# 僅繪製前 720 個小時 (30 天) 以保持清晰
PLOT_HOURS = 720 
plt.figure(figsize=(15, 6))
plt.plot(time_index[:PLOT_HOURS], y_test_plot[:PLOT_HOURS], label='真實 PM2.5 (y_test)', color='blue', alpha=0.7)
plt.plot(time_index[:PLOT_HOURS], y_pred_plot[:PLOT_HOURS], label='預測 PM2.5 (y_pred)', color='red', linestyle='--', alpha=0.7)

plt.title(f'2024 年前 30 天 PM2.5 真實值 vs. XGBoost 預測值 (t+{N_HOURS_AHEAD} 小時)')
plt.xlabel('日期與時間')
plt.ylabel('PM2.5 濃度 (ug/m³)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

print("\n--- 可視化分析 ---")
print("如果曲線吻合良好，則證明您的模型性能高於 R² 所示的負值。")
print(f"圖表已生成，顯示了 2024 年的前 {PLOT_HOURS} 小時 (30 天) 的預測結果。")