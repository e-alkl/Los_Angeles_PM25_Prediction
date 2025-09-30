import pandas as pd
import numpy as np

# Load the clean combined dataset
try:
    df = pd.read_csv("la_pm25_combined_clean.csv", index_col='DateTime', parse_dates=True)
except FileNotFoundError:
    print("❌ 錯誤：找不到 la_pm25_combined_clean.csv 文件。請確認已成功執行數據清洗腳本。")
    exit()

print(f"原始數據記錄數: {len(df)}")

# --- 1. 創建滯後特徵 (Lagged Features) ---

# PM2.5 滯後特徵 (24 小時和 1 小時)
df['PM25_Lag_24'] = df['PM25'].shift(24)
df['PM25_Lag_1'] = df['PM25'].shift(1)

# 氣象滯後特徵 (假設 6 小時前的氣象數據對現在的 PM2.5 有影響)
df['Temp_Lag_6'] = df['HourlyDryBulbTemperature'].shift(6)
df['Humidity_Lag_6'] = df['HourlyRelativeHumidity'].shift(6)
df['WindSpeed_Lag_6'] = df['HourlyWindSpeed'].shift(6)

# --- 2. 創建移動平均特徵 (Moving Averages) ---

# 過去 24 小時的 PM2.5 平均值
# 使用 .shift(1) 是為了確保計算的是「之前」的平均值，避免數據洩漏 (data leakage)
df['PM25_Avg_24'] = df['PM25'].rolling(window=24).mean().shift(1)

# --- 3. 創建時間周期特徵 (Temporal Features) ---

df['Hour'] = df.index.hour
df['DayOfWeek'] = df.index.dayofweek
df['Month'] = df.index.month # 季節性影響

# --- 4. 最終清理 ---

# 刪除因為 shift() 和 rolling() 操作而產生的 NaN 缺失值
# 由於我們用了 24 小時滯後，前 24 行會被刪除。
df.dropna(inplace=True)

# 確保所有特徵都是數值型
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\n--- 特徵工程摘要 ---")
print(f"特徵創建後記錄數: {len(df)}")
print("數據集包含的特徵:")
print(df.columns.tolist())
print("\n前五行數據:")
print(df.head())

# 保存最終特徵集
df.to_csv("la_pm25_feature_set.csv")
print("\n✅ 特徵集已保存至 la_pm25_feature_set.csv，準備模型訓練！")