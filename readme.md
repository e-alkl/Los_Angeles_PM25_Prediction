# 🏙️ Los Angeles PM2.5 Hourly Prediction (Random Forest Regression)

## Project Overview

This project focuses on **predicting the next hour's $\text{PM}_{2.5}$ concentration** in Los Angeles, California, using a combination of historical air quality data, time-series features, and localized meteorological conditions.

The goal was to develop a robust machine learning model capable of capturing the complex, cyclical, and lagged dependencies of $\text{PM}_{2.5}$ concentrations.

## Key Results & Performance

By leveraging detailed feature engineering, particularly focusing on the previous hour's $\text{PM}_{2.5}$ reading and time-based cyclical features, the model achieved exceptional performance on this short-term prediction task.

| Metric | Result |
| :--- | :--- |
| **Model** | Random Forest Regressor |
| **Coefficient of Determination ($\text{R}^2$)** | **0.9868** |
| **Root Mean Squared Error ($\text{RMSE}$)** | 0.34 $\mu g/m^3$ |

This result confirms the model's high accuracy in predicting very short-term (1-hour ahead) $\text{PM}_{2.5}$ fluctuations.

## Technology Stack

* **Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (`RandomForestRegressor`)
* **Visualization:** Matplotlib, Seaborn

## Feature Engineering Highlights

The high performance was largely attributed to identifying and leveraging key features:

1.  **Lagged $\text{PM}_{2.5}$ Feature:** The most critical input was the $\text{PM}_{2.5}$ concentration from the immediate **previous hour ($\text{PM25\_Lag\_1}$)**, which showed over $99\%$ feature importance, demonstrating the strong temporal correlation in air quality.
2.  **Cyclical Time Features:** Sinusoidal and Cosine transformations were applied to the **Hour of Day** to enable the model to capture the daily, cyclical pattern of pollution (e.g., morning and evening traffic peaks).
3.  **Meteorological Inputs:** Features like `HourlyDryBulbTemperature`, `HourlyWindSpeed`, and `HourlyAtmosphericPressure` were incorporated to account for environmental stability and dispersion.

## Visualizations

The project includes key visualizations illustrating the model's accuracy and feature contribution:

* **Prediction vs. Actual Plot:** (See `pm25_prediction_vs_actual.png`) A plot showing the near-perfect alignment between predicted and actual $\text{PM}_{2.5}$ values.
* **Feature Importance:** (See `feature_importance.png`) A chart confirming the dominant role of the lagged $\text{PM}_{2.5}$ feature.

## Future Work & Optimization

The current model excels at short-term prediction. To increase the project's complexity and practical value, future work will focus on the following:

1.  **Deeper Prediction Challenge:** Re-engineering the target variable to predict **6-hour** and **24-hour** ahead $\text{PM}_{2.5}$ concentrations. This will force the model to rely less on the highly correlated lagged feature and more on meteorological and time-based inputs.
2.  **Model Robustness:** Implementing **Time Series Cross-Validation** to rigorously test the model's performance across different time periods.
3.  **Explainability Analysis ($\text{SHAP}$):** Using $\text{SHAP}$ (SHapley Additive exPlanations) values to interpret model decisions, especially to understand which meteorological factors drive predictions during periods of high pollution.
4.  **Wind Direction Encoding:** Correctly encoding the cyclical nature of wind direction using $\text{sin/cos}$ transformations for improved accuracy.