import pandas as pd
import numpy as np
import os

# --- 1. File and Column Definitions (English/Simplified) ---
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
PM25_FILE_PATTERN = 'hourly_88101_{year}.csv'
LCD_FILE_PATTERN = 'LCD_USW00093134_{year}.csv'
OUTPUT_FILE = 'la_pm25_combined_clean_v15_2019_2024.csv'

# Final column names
PM25_COLUMN = 'Sample Measurement'
LCD_COLUMNS = [
    'HourlyDryBulbTemperature', 'HourlyRelativeHumidity', 
    'HourlyWindSpeed', 'HourlyDewPointTemperature', 
    'HourlyStationPressure', 'HourlySeaLevelPressure', 
    'HourlyVisibility'
]

# --- Helper Functions ---
def load_and_clean_pm25(year):
    """Loads and cleans single year PM2.5 data."""
    file_path = PM25_FILE_PATTERN.format(year=year)
    if not os.path.exists(file_path):
        print(f"🚨 Warning: PM2.5 file {file_path} not found. Skipping year.")
        return None
        
    df = pd.read_csv(file_path, parse_dates=['Date GMT'], low_memory=False)
    
    df = df.rename(columns={'Date GMT': 'DateTime'})
    df = df.set_index('DateTime')
    df = df[[PM25_COLUMN]].rename(columns={PM25_COLUMN: 'PM25'})
    
    # Clean: Negative values are invalid measurements
    df.loc[df['PM25'] < 0, 'PM25'] = np.nan
    df = df[~df.index.duplicated(keep='first')] # Remove duplicate time indices
    return df

def load_and_clean_lcd(year):
    """Loads and cleans single year meteorological (LCD) data."""
    file_path = LCD_FILE_PATTERN.format(year=year)
    if not os.path.exists(file_path):
        print(f"🚨 Warning: LCD file {file_path} not found. Skipping year.")
        return None

    df = pd.read_csv(file_path, parse_dates=['DATE'], low_memory=False)

    df = df.rename(columns={'DATE': 'DateTime'})
    df = df.set_index('DateTime')
    df = df[LCD_COLUMNS].copy()

    # Clean: Convert to numeric, coerce errors, and clean extreme/placeholder values
    for col in LCD_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Common placeholder for missing/bad data in weather files: -9999
        if col in ['HourlyDryBulbTemperature', 'HourlyDewPointTemperature']:
             # Temperature data: Assume anything below -50 is an error/placeholder
             df.loc[df[col] < -50, col] = np.nan
        # Visibility: Values of 0 can be placeholders or bad measurements, often better to treat as missing
        if col == 'HourlyVisibility':
             df.loc[df[col] <= 0, col] = np.nan
             
    df = df[~df.index.duplicated(keep='first')]
    return df

# --- 2. Main Cleaning and Merging Pipeline ---
if __name__ == "__main__":
    
    all_pm25 = []
    all_lcd = []
    
    print("--- Starting to load and clean 2019-2024 data ---")
    
    # Load and clean data year by year
    for year in YEARS:
        print(f"Processing year: {year}")
        pm25_df = load_and_clean_pm25(year)
        if pm25_df is not None:
            all_pm25.append(pm25_df)
            
        lcd_df = load_and_clean_lcd(year)
        if lcd_df is not None:
            all_lcd.append(lcd_df)

    # Concatenate all years
    combined_pm25 = pd.concat(all_pm25, axis=0).sort_index()
    combined_lcd = pd.concat(all_lcd, axis=0).sort_index()
    
    combined_pm25 = combined_pm25[~combined_pm25.index.duplicated(keep='first')]
    combined_lcd = combined_lcd[~combined_lcd.index.duplicated(keep='first')]
    print("✅ Successfully removed duplicate time indices after cross-year concatenation.")

    # Create a full time series index (hourly frequency)
    # Ensure data frames are not empty before getting min/max index
    if combined_lcd.empty or combined_pm25.empty:
        print("🚨 Error: One or both datasets are empty after loading.")
        exit()

    start_date = min(combined_pm25.index.min(), combined_lcd.index.min())
    end_date = max(combined_pm25.index.max(), combined_lcd.index.max())
    full_index = pd.date_range(start=start_date, end=end_date, freq='h', name='DateTime') 

    # Reindex (Align PM2.5 and LCD data to the full time series)
    combined_pm25 = combined_pm25.reindex(full_index)
    combined_lcd = combined_lcd.reindex(full_index)

    print(f"PM2.5 total records (re-indexed): {len(combined_pm25)}")
    print(f"LCD total records (re-indexed): {len(combined_lcd)}")
    
    # Horizontal merge (Outer Join)
    final_df = combined_pm25.join(combined_lcd, how='outer') 
    print(f"\n--- Total records after outer join: {len(final_df)} ---")
    
    # Critical: Enforce float64 type for all numerical columns
    numeric_cols = ['PM25'] + LCD_COLUMNS
    for col in numeric_cols:
        final_df[col] = final_df[col].astype('float64')
    
    # Impute missing values using ffill/bfill combination
    initial_nan_count = final_df.isnull().sum().sum()
    final_df = final_df.fillna(method='ffill')
    final_df = final_df.fillna(method='bfill')
    
    # Check for remaining NaNs (only possible if the very start or end of the series had full NaNs)
    final_nan_count = final_df.isnull().sum().sum()
    
    # Final dataset check
    final_count = len(final_df)
    print(f"Final clean records: {final_count}")
    print(f"NaNs imputed: {initial_nan_count - final_nan_count}")
    
    # Save final result
    if final_count > 0 and final_nan_count == 0:
        final_df.to_csv(OUTPUT_FILE)
        print(f"\n✅ Successfully saved {final_count} records to: {OUTPUT_FILE}")
    elif final_nan_count > 0:
        print
