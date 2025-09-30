import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib

# --- 1. 數據載入與特徵定義 ---
# 載入 V13 清洗完成的數據集
file_path = "la_pm25_combined_clean_v13.csv"
try:
    # 🚨 確保這裡有 df 變數的定義
    df = pd.read_csv(file_path, index_col='DateTime', parse_dates=True)
except FileNotFoundError:
    print(f"🚨 錯誤：找不到文件 {file_path}。請確保它與腳本在同一目錄下。")
    exit()

print("--- 模型訓練準備階段 ---")
print(f"載入數據集總記錄數: {len(df)}")

# 定義目標變量 (y)
target_variable = 'PM25'
y = df[target_variable]

# 定義氣象特徵 (X)
feature_columns = [
    'HourlyDryBulbTemperature', 
    'HourlyRelativeHumidity', 
    'HourlyWindSpeed', 
    'HourlyDewPointTemperature', 
    'HourlyStationPressure', 
    'HourlySeaLevelPressure', 
    'HourlyVisibility'
]
X = df[feature_columns].copy()

# --- 1.5. 新增：目標變量的滯後特徵 (Lagged Feature) ---
# 這是提升 R^2 的關鍵。
X['PM25_Lag_1'] = y.shift(1)

# 由於第一個小時 (t=0) 沒有 t-1 的值，會產生 NaN。
# 我們需要刪除第一個小時的記錄，以確保數據完整性。
X.dropna(inplace=True) 
y = y[X.index] # 確保 y 也被同步截斷，保持嚴格對齊

print(f"新增 PM2.5 滯後特徵後，總記錄數變為: {len(X)}")

# --- 2. 創建時間相關特徵 (特徵工程) ---
print("⚙️ 進行時間特徵工程...")

# 從索引提取時間特徵
X['DayOfWeek'] = X.index.dayofweek
X['Month'] = X.index.month

# 💡 增加時間的循環特徵
X['Hour'] = X.index.hour
X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
X = X.drop('Hour', axis=1) # 移除原始的 Hour 欄位

print(f"最終特徵 (X) 數量: {X.shape[1]}")


# --- 3. 數據集劃分 (隨機劃分：80% 訓練 / 20% 測試) ---
print("\n⚙️ 進行隨機劃分：80% 訓練 / 20% 測試...")

# 使用 train_test_split 確保 X 和 y 完美對齊
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"訓練集 (80%) 記錄數: {len(X_train)}")
print(f"測試集 (20%) 記錄數: {len(X_test)}")


# --- 4. 模型訓練 (隨機森林回歸) ---
print("\n⚙️ 開始訓練隨機森林回歸模型 (V2.0)...")
# 保持 max_depth=10 以控制模型複雜度
model = RandomForestRegressor(n_estimators=100, 
                              max_depth=10, 
                              random_state=42, 
                              n_jobs=-1)
model.fit(X_train, y_train)
print("✅ 模型訓練完成！")


# --- 5. 模型評估 ---
y_pred = model.predict(X_test)

# 計算評估指標
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- 模型性能評估 (V2.0 測試集) ---")
print(f"平均絕對誤差 (RMSE): {rmse:.2f}")
print(f"決定係數 (R²): {r2:.4f}")


# --- 6. 特徵重要性分析 ---
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\n--- 頂部 5 個重要特徵 (滯後特徵影響分析) ---")
print(feature_importances.nlargest(5))

# --- 7. 模型儲存 (Save Model) ---
model_filename = 'pm25_random_forest_v2.0.joblib'

try:
    # 使用 joblib 儲存訓練好的模型 (model 變數)
    joblib.dump(model, model_filename)
    print(f"\n✅ 模型已成功儲存至 {model_filename}！")
except Exception as e:
    print(f"\n🚨 模型儲存失敗: {e}")

# --- 8. 視覺化設置 ---
print("\n--- 7. 繪製視覺化圖表 ---")
plt.style.use('seaborn-v0_8-whitegrid') # 設置圖表風格
sns.set_palette("colorblind") # 設置顏色主題


# --- 圖表 1: 預測與實際值對比 (時間序列) ---
# 選擇測試集 (20% 數據) 中一段代表性的時間範圍來展示高準確度
# 這裡我們隨機選擇測試集中的 24*7 = 168 個連續小時 (一周)
n_hours = 168
start_index = np.random.randint(0, len(y_test) - n_hours)

# 創建一個 DataFrame 包含實際值和預測值
plot_df = pd.DataFrame({
    'Actual PM2.5': y_test.iloc[start_index : start_index + n_hours],
    'Predicted PM2.5': y_pred[start_index : start_index + n_hours]
})

plt.figure(figsize=(15, 6))
plt.plot(plot_df.index, plot_df['Actual PM2.5'], label='Actual PM2.5', color='blue', linewidth=2)
plt.plot(plot_df.index, plot_df['Predicted PM2.5'], label='Predicted PM2.5', color='red', linestyle='--', linewidth=1.5)

plt.title(f'PM2.5 預測與實際值對比 (一週樣本)\n (RMSE: {rmse:.2f} | R²: {r2:.4f})', fontsize=16)
plt.xlabel('時間 (DateTime)', fontsize=12)
plt.ylabel(r'PM$_{2.5}$ $(\mu g/m^3)$', fontsize=12)
plt.legend(fontsize=10)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('pm25_prediction_vs_actual.png')
print("✅ 圖表 1: 預測與實際值對比 (pm25_prediction_vs_actual.png) 已保存")
# plt.show() # 如果在 Jupyter/Colab 中運行，取消註釋此行


# --- 圖表 2: 特徵重要性分析 (Feature Importance) ---
# 使用你在 Section 6 計算的 feature_importances
# 選擇前 10 個最重要的特徵來展示 (PM25_Lag_1 會佔據絕對優勢)
top_n = feature_importances.nlargest(10).sort_values(ascending=True)

plt.figure(figsize=(10, 7))
top_n.plot(kind='barh', color='darkgreen')

plt.title('特徵重要性分析 (Feature Importance)', fontsize=16)
plt.xlabel('重要性分數 (Normalized Score)', fontsize=12)
plt.ylabel('特徵名稱', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ 圖表 2: 特徵重要性分析 (feature_importance.png) 已保存")
# plt.show() # 如果在 Jupyter/Colab 中運行，取消註釋此行

print("\n🎉 成果視覺化完成！請檢查同目錄下的圖片檔案。")