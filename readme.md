Los Angeles PM2.5 Concentration 24-Hour Prediction Model (XGBoost Optimized)
Key Highlights
This project utilizes an XGBoost regressor model, integrating meteorological data and time-series features to achieve accurate prediction of PM 
2.5
​
  concentrations 24 hours in advance for the Los Angeles area.

Core Achievement: Model accuracy was significantly improved by switching the loss function, reducing the Root Mean Squared Error (RMSE) from 5.69μg/m 
3
  to a remarkable 4.50μg/m 
3
 . This represents an over 21% improvement in predictive accuracy.

Physical Interpretability: The model shifted its focus from simple historical PM 
2.5
​
  inertia to Hourly Dew Point Temperature as its most important predictor, correctly capturing the physics of atmospheric stability and pollutant accumulation.

Bias Correction: Successfully eliminated the systematic overestimation bias that the initial model exhibited during low PM 
2.5
​
  concentration periods (below 5μg/m 
3
 ).

⚙️ Model and Technology Stack (Stack)
Model Architecture: XGBoost Regressor

Optimization Strategy: Switched objective to reg:absoluteerror (MAE) and fine-tuned hyperparameters (n_estimators=200,max_depth=7).

Key Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, SHAP

Language: Python

Data and Feature Engineering
The model was trained on historical data from early 2019 through the end of 2023 and tested on data from 2024.

Feature Type	Example Feature	Description
Pollutant Lag	PM25_Lag_24	PM 
2.5
​
  concentration from 24 hours prior.
Core Meteorology	HourlyDewPointTemperature	The most critical feature in the optimized model, indicating moisture content and atmospheric stability.
Atmospheric Stability	HourlyStationPressure	Suggests high-pressure ridges or inversion conditions, critical for trapping pollutants.
Periodicity	Month, Hour_sin/cos	Captures seasonal (e.g., winter pollution) and diurnal (hourly) trends.

Final Results and Analysis (Results)
1. Performance Metric Comparison (2024 Test Set)
Model Version	Loss Function	RMSE (μg/m 
3
 )	R²
Initial Version	MSE	5.69	−0.87
Final Optimized	MAE	4.50	−0.17
2. Model Interpretability (SHAP Analysis)
The plot below illustrates how Dew Point Temperature influences the PM 
2.5
​
  prediction: higher dew point leads to a higher predicted PM 
2.5
​
  due to increased atmospheric moisture and stability.

3. Prediction Curve (Optimized Model)
The curve shows that the optimized model's predictions (red dashed line) align very closely with the true values (blue solid line), successfully eliminating the over-prediction bias in low-concentration periods.

How to Run (Getting Started)
1. Data and Model Acquisition
Due to file size limits, data and the final model file are excluded via .gitignore. To reproduce the results, you need to:

Obtain the raw CSV data file (consistent with the format expected in pm25_model_train.py).

Run the pm25_model_train.py script to train the model and generate the required pm25_xgb_optimized_predict_24hr_2019_2023_train.joblib file.

2. Installation
Install all necessary Python libraries in your environment:

Bash

pip install pandas numpy xgboost scikit-learn matplotlib shap joblib
3. Running Analysis and Prediction
Execute the relevant Python scripts:

Bash

# To train/retrain the final optimized model
python pm25_model_train.py

# To generate the prediction curve plot (requires updating MODEL_FILENAME in the script)
python pm25_plot_results.py

# To generate the SHAP analysis plots
python pm25_shap_plot.py
