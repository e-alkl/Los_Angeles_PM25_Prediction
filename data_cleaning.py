import pandas as pd
import numpy as np
from datetime import timedelta 

# --- 0. 文件列表與時區設定 ---
# 🚨 請確保這四個 CSV 檔案都在腳本的同一目錄下！
pm25_files = ["hourly_88101_2023.csv", "hourly_88101_2024.csv"] 
weather_files = ["LCD_USW00093134_2023.csv", "LCD_USW00093134_2024.csv"] 
LA_TIMEZONE = 'America/Los_Angeles'

print("--- V13 數據清洗與合併：時間對齊、索引去重、變數定義修正 ---")

# --- 1. PM2.5 數據合併與清洗 (使用 Local 時間) ---
df_pm_list = []
for file in pm25_files:
    try:
        df = pd.read_csv(file, low_memory=False)
        
        # 選擇 Local 時間欄位和 PM2.5 值
        df_pm_clean = df[['Date Local', 'Time Local', 'Sample Measurement']].copy()
        df_pm_clean.columns = ['Date', 'Time', 'PM25']
        
        # 創建 Datetime 索引 (視為無時區的當地時間)
        df_pm_clean['DateTime'] = pd.to_datetime(df_pm_clean['Date'] + ' ' + df_pm_clean['Time'])
        df_pm_clean['PM25'] = pd.to_numeric(df_pm_clean['PM25'], errors='coerce')
        df_pm_clean.dropna(subset=['PM25'], inplace=True) 
        
        # 確保時間戳格式為標準小時開始時間 (使用 'h' 避免 FutureWarning)
        df_pm_clean['DateTime'] = df_pm_clean['DateTime'].dt.floor('h') 
        
        # 處理 PM2.5 數據中的重複時間戳 (例如，DST 期間的重複測量)
        df_pm_clean = df_pm_clean.groupby('DateTime')['PM25'].mean().reset_index()

        df_hourly_avg = df_pm_clean.set_index('DateTime')
        
        df_pm_list.append(df_hourly_avg)
        print(f"✅ 成功處理 {file}，取得 {len(df_hourly_avg)} 條小時平均記錄")

    except Exception as e:
        print(f"❌ 錯誤處理 {file}: {e}")
        continue

df_pm_combined = pd.concat(df_pm_list, ignore_index=False)
# 最終 PM2.5 合併時也要去重，以防文件邊界重複
df_pm_combined = df_pm_combined[~df_pm_combined.index.duplicated(keep='first')] 
print(f"\nPM2.5 兩年總計小時平均記錄數: {len(df_pm_combined)}")


# --- 2. NOAA 氣象數據合併與清洗（強制當地時間與偏移 & 索引去重） ---
# 🚨 變數初始化，解決 NameError: name 'df_weather_list' is not defined
df_weather_list = [] 
weather_features = [
    'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyWindSpeed', 
    'HourlyDewPointTemperature', 'HourlyStationPressure', 'HourlySeaLevelPressure', 
    'HourlyVisibility', 'HourlyWindDirection', 'HourlyWindGustSpeed', 
    'HourlyPrecipitation',
]

for file in weather_files:
    try:
        df = pd.read_csv(file, low_memory=False)
        
        # 處理 NOAA 數據的 ISO 格式
        df['DateTime'] = pd.to_datetime(df['DATE'])
        
        # 將 NOAA 的小時結束時間偏移到小時開始時間 (例如 01:00 變為 00:00)
        df['DateTime'] = df['DateTime'] - pd.Timedelta(hours=1) 
        
        # 移除任何可能的時區和秒數/毫秒
        df['DateTime'] = df['DateTime'].dt.tz_localize(None).dt.floor('h')
        
        df = df.set_index('DateTime')
        
        # 🚨 V13 關鍵修正: 移除重複的時間索引，解決 cannot reindex 錯誤！
        df = df[~df.index.duplicated(keep='first')]

        # 強制重新採樣索引，以確保每小時都有記錄
        if not df.empty:
            start_date = df.index.min()
            end_date = df.index.max()
            full_hourly_index = pd.date_range(start=start_date, end=end_date, freq='h')
            df = df.reindex(full_hourly_index)
            df.index.name = 'DateTime' 
        
        # 選擇特徵並轉為數值
        df = df[weather_features].copy()
        for col in weather_features:
            # 將非數值標記替換為 NaN
            df[col] = df[col].replace(['M', 'T'], np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df_weather_list.append(df)
        print(f"✅ 成功載入 {file} ({len(df)} 條記錄) - 索引已強制連續化。")

    except Exception as e:
        print(f"❌ 錯誤處理 {file}: {e}")
        continue

# 🚨 變數定義，解決 NameError: name 'df_weather_combined' is not defined
df_weather_combined = pd.concat(df_weather_list, ignore_index=False)
df_weather_combined = df_weather_combined[~df_weather_combined.index.duplicated(keep='first')]
print(f"NOAA 兩年總計小時記錄數（含 NaN 佔位）： {len(df_weather_combined)}")


# --- 3. 合併、清理並保存 ---
# 🚨 使用 left join 以 PM2.5 數據為基礎
df_combined = df_pm_combined.join(df_weather_combined, how='left')

df_combined.replace([9999.0, 999.0, 9999, 'M', 'T'], np.nan, inplace=True) 

# --- 數據插值：填充缺失的氣象數據 ---
print("\n⚙️ 正在插入缺失的氣象數據 (使用 Forward Fill)...")
# 使用前一個有效的觀測值來填補缺失的小時
df_combined[weather_features] = df_combined[weather_features].fillna(method='ffill')

# 最終丟棄 PM2.5 或關鍵氣象數據仍然缺失的行
df_combined = df_combined.dropna(subset=['PM25', 'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 'HourlyWindSpeed'])

print("\n--- 兩年合併數據集摘要 ---")
print(f"數據範圍: {df_combined.index.min()} 到 {df_combined.index.max()}")
print(f"最終用於模型的記錄數: {len(df_combined)}")

output_filename = "la_pm25_combined_clean_v13.csv"
df_combined.to_csv(output_filename)
print(f"\n✅ V13 清洗數據已保存至 {output_filename}！")