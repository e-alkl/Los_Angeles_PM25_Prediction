import pandas as pd
import numpy as np

# --- 1. 檔案定義：涵蓋 2019 到 2024 年 ---
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
PM25_FILE_PATTERN = 'hourly_88101_{year}.csv'
LCD_FILE_PATTERN = 'LCD_USW00093134_{year}.csv'
OUTPUT_FILE = 'la_pm25_combined_clean_v15_2019_2024.csv'

# 最終 PM2.5 欄位名稱
PM25_COLUMN = 'Sample Measurement'

# 最終氣象欄位名稱
LCD_COLUMNS = [
    'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 
    'HourlyWindSpeed', 'HourlyDewPointTemperature', 
    'HourlyStationPressure', 'HourlySeaLevelPressure', 
    'HourlyVisibility'
]

# ... [ load_and_clean_pm25 和 load_and_clean_lcd 函式保持不變，略 ]
def load_and_clean_pm25(year):
    """加載並清洗單一年份的 PM2.5 數據。"""
    file_path = PM25_FILE_PATTERN.format(year=year)
    try:
        df = pd.read_csv(file_path, parse_dates=['Date GMT'], low_memory=False)
    except FileNotFoundError:
        print(f"🚨 警告: 找不到 PM2.5 檔案 {file_path}，跳過此年份。")
        return None
    
    df = df.rename(columns={'Date GMT': 'DateTime'})
    df = df.set_index('DateTime')
    df = df[[PM25_COLUMN]].rename(columns={PM25_COLUMN: 'PM25'})
    df.loc[df['PM25'] < 0, 'PM25'] = np.nan
    df = df[~df.index.duplicated(keep='first')] 
    return df

def load_and_clean_lcd(year):
    """加載並清洗單一年份的氣象數據。"""
    file_path = LCD_FILE_PATTERN.format(year=year)
    try:
        df = pd.read_csv(file_path, parse_dates=['DATE'], low_memory=False)
    except FileNotFoundError:
        print(f"🚨 警告: 找不到 LCD 檔案 {file_path}，跳過此年份。")
        return None

    df = df.rename(columns={'DATE': 'DateTime'})
    df = df.set_index('DateTime')
    df = df[LCD_COLUMNS].copy()

    for col in LCD_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[~df.index.duplicated(keep='first')]
    return df

# --- 2. 主清洗與合併流程 ---
if __name__ == "__main__":
    
    all_pm25 = []
    all_lcd = []
    
    print("--- 開始加載和清洗 2019-2024 年數據 ---")
    
    for year in YEARS:
        print(f"處理年份: {year}")
        pm25_df = load_and_clean_pm25(year)
        if pm25_df is not None:
            all_pm25.append(pm25_df)
        lcd_df = load_and_clean_lcd(year)
        if lcd_df is not None:
            all_lcd.append(lcd_df)

    combined_pm25 = pd.concat(all_pm25, axis=0).sort_index()
    combined_lcd = pd.concat(all_lcd, axis=0).sort_index()
    
    combined_pm25 = combined_pm25[~combined_pm25.index.duplicated(keep='first')]
    combined_lcd = combined_lcd[~combined_lcd.index.duplicated(keep='first')]
    print("✅ 成功移除跨年合併後產生的重複時間索引。")

    # 創建一個完整的時間序列索引
    start_date = combined_lcd.index.min()
    end_date = combined_lcd.index.max()
    full_index = pd.date_range(start=start_date, end=end_date, freq='h', name='DateTime') 

    # 將 PM2.5 和 LCD 數據對齊到完整的時間序列
    combined_pm25 = combined_pm25.reindex(full_index)
    combined_lcd = combined_lcd.reindex(full_index)

    print(f"PM2.5 總記錄數 (對齊後): {len(combined_pm25)}")
    print(f"LCD 總記錄數 (對齊後): {len(combined_lcd)}")
    
    # 水平合併 PM2.5 和 LCD 數據
    final_df = combined_pm25.join(combined_lcd, how='outer') 
    print(f"\n--- 合併後 (outer join) 總記錄數: {len(final_df)} ---")
    
    # 關鍵修正：強制將所有數值欄位轉換為 float64
    numeric_cols = ['PM25'] + LCD_COLUMNS
    for col in numeric_cols:
        final_df[col] = final_df[col].astype('float64')
    
    # 填充：使用 ffill/bfill 組合
    final_df = final_df.fillna(method='ffill')
    final_df = final_df.fillna(method='bfill')


    # 🚨 關鍵改變：移除 final_df.dropna(inplace=True)，讓訓練腳本去處理殘餘的 NaN
    # final_df.dropna(inplace=True) 

    # 最終數據集檢查 (這次會有少數 NaN)
    final_count = len(final_df)
    missing_count = final_df.isnull().sum().sum()
    print(f"最終清洗完成的記錄數: {final_count}")
    print(f"缺失值總和 (預計為少量，但在訓練腳本中會被清除): {missing_count}")
    
    # 儲存最終結果
    if final_count > 0:
        final_df.to_csv(OUTPUT_FILE)
        print(f"\n✅ 成功保存 {final_count} 筆記錄至：{OUTPUT_FILE}")
    else:
        print("\n🚨 警告：數據保存失敗，請檢查原始 CSV 文件。")