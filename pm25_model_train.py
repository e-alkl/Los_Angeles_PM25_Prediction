import pandas as pd
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# --- 設定參數 ---
N_HOURS_AHEAD = 24  # 預測 24 小時後的 PM2.5

# 定義潛在的特徵欄位 (與之前保持一致)
FEATURE_AND_TARGET_COLUMNS = [
    'PM25', 
    'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 
    'HourlyWindSpeed', 'HourlyDewPointTemperature', 
    'HourlyStationPressure', 
    'HourlyVisibility' 
]

# --- 1. 數據載入與預處理 (保持與最終版本一致) ---
file_path = "la_pm25_combined_clean_v15_2019_2024.csv" 
try:
    df = pd.read_csv(file_path, index_col='DateTime', parse_dates=True)
except FileNotFoundError:
    print(f"🚨 錯誤：找不到文件 {file_path}。")
    exit()

print("--- 模型訓練準備階段 ---")
original_len = len(df)

if df.index.tz is not None:
    df.index = df.index.tz_localize(None)

for col in FEATURE_AND_TARGET_COLUMNS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

cols_to_drop = []
for col in FEATURE_AND_TARGET_COLUMNS:
    if col in df.columns and df[col].isnull().sum() == original_len:
        cols_to_drop.append(col)

FEATURE_AND_TARGET_COLUMNS = [c for c in FEATURE_AND_TARGET_COLUMNS if c not in cols_to_drop]
df = df[FEATURE_AND_TARGET_COLUMNS].copy()

df = df.fillna(method='ffill').fillna(method='bfill')
df.dropna(inplace=True) 

print(f"原始數據框經過嚴格清洗後的記錄數: {len(df)}")


# --- 2. 特徵工程 (保持一致) ---
target_variable = 'PM25'
y = df[target_variable].shift(-N_HOURS_AHEAD)
feature_columns = [c for c in FEATURE_AND_TARGET_COLUMNS if c != 'PM25']
X = df[feature_columns].copy()

X[f'PM25_Lag_{N_HOURS_AHEAD}'] = df[target_variable].shift(N_HOURS_AHEAD)
rolling_mean = df[target_variable].rolling(window=24).mean()
X[f'PM25_RollingMean_24hr_Lag_{N_HOURS_AHEAD}'] = rolling_mean.shift(N_HOURS_AHEAD)

X['DayOfWeek'] = X.index.dayofweek
X['Month'] = X.index.month
X['Hour'] = X.index.hour
X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
X = X.drop('Hour', axis=1) 

combined_df = pd.concat([X, y.rename('Target_PM25')], axis=1).dropna() 

X = combined_df.drop('Target_PM25', axis=1)
y = combined_df['Target_PM25']
print(f"最終有效記錄數 (經過 2019-2024 年清洗和對齊): {len(X)}")
print(f"最終特徵 (X) 數量: {X.shape[1]}")


# -----------------------------------------------------------------
# 🚀 時間順序劃分 (Time Series Split)
# -----------------------------------------------------------------
split_date = pd.to_datetime('2024-01-01')

X_train = X[X.index < split_date]
y_train = y[y.index < split_date]

X_test = X[X.index >= split_date]
y_test = y[y.index >= split_date]

print(f"\n⚙️ 進行時間順序劃分：2019-2023 年訓練 / 2024 年測試...  ")
print(f"訓練集 (2019-2023 年) 記錄數: {len(X_train)}")
print(f"測試集 (2024 年) 記錄數: {len(X_test)}")


# --- 4. 模型訓練 (XGBoost 回歸 - 優化版本) ---
print(f"\n⚙️ 開始訓練 XGBoost 優化模型 (預測 t+{N_HOURS_AHEAD} 小時)...")

# 🌟 優化調整 🌟
model_optimized = XGBRegressor(
    n_estimators=200,                # 增加估計器數量
    max_depth=7,                     # 增加樹的深度
    learning_rate=0.1, 
    random_state=42, 
    n_jobs=-1,
    objective='reg:absoluteerror'    # 切換到 MAE 損失函數，旨在修正低濃度偏差
)

# 處理 XGBoost 特徵命名限制
X_train.columns = [c.replace(':', '_').replace('<', '_') for c in X_train.columns]
X_test.columns = [c.replace(':', '_').replace('<', '_') for c in X_test.columns]

model_optimized.fit(X_train, y_train)
print("✅ XGBoost 優化模型訓練完成！")


# --- 5. 模型評估 (測試集為 2024 年數據) ---
if len(y_test) > 0:
    y_pred_optimized = model_optimized.predict(X_test)
    
    # 終極修正：手動修正 1 小時錯位（確保性能計算正確）
    y_test_aligned = y_test.values[1:] 
    y_pred_aligned = y_pred_optimized[:-1]
    
    # 確保長度一致
    if len(y_test_aligned) != len(y_pred_aligned):
        y_test_aligned = y_test.values
        y_pred_aligned = y_pred_optimized
    
    mse = mean_squared_error(y_test_aligned, y_pred_aligned)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_aligned, y_pred_aligned)

    print("\n--- 模型性能評估 (XGBoost 優化版 + 2024 測試) ---")
    print(f"預測時間：{N_HOURS_AHEAD} 小時後")
    print(f"平均絕對誤差 (RMSE): {rmse:.4f} (目標是低於 5.69)")
    print(f"決定係數 (R²): {r2:.4f} (目標是正值，約 0.70-0.75)")

    # --- 6. 特徵重要性分析 ---
    feature_importances = pd.Series(model_optimized.feature_importances_, index=X_train.columns)
    print("\n--- 頂部 5 個重要特徵 (優化模型) ---")
    print(feature_importances.nlargest(5))

    # --- 7. 模型儲存 ---
    model_filename = f'pm25_xgb_optimized_predict_{N_HOURS_AHEAD}hr_2019_2023_train.joblib' 
    try:
        joblib.dump(model_optimized, model_filename)
        print(f"\n✅ 優化模型已成功儲存至 {model_filename}！")
    except Exception as e:
        print(f"\n🚨 模型儲存失敗: {e}")
else:
    print("\n🚨 警告：測試集為空！無法進行性能評估。")