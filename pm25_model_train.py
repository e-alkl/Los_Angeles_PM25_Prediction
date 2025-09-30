import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import joblib
import shap # 🚀 新增 SHAP 庫導入


# --- 設定優化階段參數 ---
N_HOURS_AHEAD = 24 


# --- 1. 數據載入與特徵定義 ---
file_path = "la_pm25_combined_clean_v13.csv"
try:
    df = pd.read_csv(file_path, index_col='DateTime', parse_dates=True)
except FileNotFoundError:
    print(f"🚨 錯誤：找不到文件 {file_path}。請確保它與腳本在同一目錄下。")
    exit()

print("--- 模型訓練準備階段 ---")
print(f"載入數據集總記錄數: {len(df)}")

# 定義目標變量 (y) - 使用 t+N 的數值
target_variable = 'PM25'
y = df[target_variable].shift(-N_HOURS_AHEAD)


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
# 🚀 關鍵優化 1: 加入 N 小時前的 PM2.5 數值
X[f'PM25_Lag_{N_HOURS_AHEAD}'] = df[target_variable].shift(N_HOURS_AHEAD)

# 🌟 關鍵優化 2: 加入 N 小時前的 24 小時滾動平均 (Rolling Mean)
rolling_mean = df[target_variable].rolling(window=24).mean()
X[f'PM25_RollingMean_24hr_Lag_{N_HOURS_AHEAD}'] = rolling_mean.shift(N_HOURS_AHEAD)
print("✅ 新增 PM2.5 過去 24 小時平均值特徵！")


# --- 2. 創建時間相關特徵 (特徵工程) ---
print("⚙️ 進行時間特徵工程...")
X['DayOfWeek'] = X.index.dayofweek
X['Month'] = X.index.month

# 💡 增加時間的循環特徵 (不變)
X['Hour'] = X.index.hour
X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
X = X.drop('Hour', axis=1) # 移除原始的 Hour 欄位


# --- 2.5. 清理 NaN 值和對齊 X 與 y (數據對齊是成功的關鍵) ---
combined_df = pd.concat([X, y.rename('Target_PM25')], axis=1).dropna()

# 重新定義 X 和 y
X = combined_df.drop('Target_PM25', axis=1)
y = combined_df['Target_PM25']

print(f"定義滯後特徵和目標變數後，最終有效記錄數: {len(X)}")
print(f"最終特徵 (X) 數量: {X.shape[1]}")


# --- 3. 數據集劃分 (隨機劃分：80% 訓練 / 20% 測試) ---
print("\n⚙️ 進行隨機劃分：80% 訓練 / 20% 測試...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"訓練集 (80%) 記錄數: {len(X_train)}")
print(f"測試集 (20%) 記錄數: {len(X_test)}")


# --- 4. 模型訓練 (隨機森林回歸) ---
print(f"\n⚙️ 開始訓練隨機森林模型 (預測 t+{N_HOURS_AHEAD} 小時)...")
model = RandomForestRegressor(n_estimators=100, 
                              max_depth=10, 
                              random_state=42, 
                              n_jobs=-1)
model.fit(X_train, y_train)
print("✅ 模型訓練完成！")


# --- 5. 模型評估 ---
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- 模型性能評估 (新挑戰測試集) ---")
print(f"預測時間：{N_HOURS_AHEAD} 小時後")
print(f"平均絕對誤差 (RMSE): {rmse:.2f}")
print(f"決定係數 (R²): {r2:.4f}")


# --- 6. 特徵重要性分析 ---
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\n--- 頂部 5 個重要特徵 (滯後特徵影響分析) ---")
print(feature_importances.nlargest(5))


# --- 7. 模型儲存 (Save Model) ---
model_filename = f'pm25_rf_predict_{N_HOURS_AHEAD}hr_v2.joblib' 
try:
    joblib.dump(model, model_filename)
    # 由於模型檔案通常很大，我們不應該將其推送到 Git，因此這裡註釋掉了關於 Git 的說明
    print(f"\n✅ 模型已成功儲存至 {model_filename}！")
except Exception as e:
    print(f"\n🚨 模型儲存失敗: {e}")


# --- 8. 視覺化設置 ---
print("\n--- 繪製視覺化圖表 ---")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind") 


# --- 圖表 1 & 2: 預測對比與傳統特徵重要性 (與之前相同，略) ---
# ... [這部分代碼與您提供的 V2 版本完全相同，因此為了簡潔這裡省略]
# ... [確保您本地腳本中這部分仍然存在]


# -----------------------------------------------------------
# 🌟 關鍵優化 3: SHAP 可解釋性分析 (Explaining the Model)
# -----------------------------------------------------------
print("\n⚙️ 執行 SHAP 分析，解釋模型決策...")

# 1. 創建 Explainer
# 由於 Random Forest 是樹模型，我們使用 TreeExplainer，速度最快、精度最高
explainer = shap.TreeExplainer(model)

# 2. 計算 SHAP 值
# 為了速度和準確性，我們只在測試集的一個子樣本上運行 SHAP (例如 500 個樣本)
X_sample = X_test.sample(n=500, random_state=42) 
shap_values = explainer.shap_values(X_sample)

# 3. 繪製 SHAP 摘要圖
# 這張圖是 SHAP 報告的核心，展示了每個特徵的影響方向和分佈
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, 
                  plot_type='dot', 
                  show=False, 
                  title=f'SHAP Feature Impact (Predicting T+{N_HOURS_AHEAD}hr PM2.5)')

# 調整佈局並保存
plt.tight_layout()
shap_filename = f'shap_summary_plot_{N_HOURS_AHEAD}hr_v2.png'
plt.savefig(shap_filename, bbox_inches='tight')
plt.close() # 關閉圖表，避免內存洩漏

print(f"✅ 圖表 3: SHAP 摘要圖 ({shap_filename}) 已保存")

print("\n🎉 成果視覺化和 SHAP 分析完成！請檢查同目錄下的圖片檔案。")
# --- 程式碼結束 ---