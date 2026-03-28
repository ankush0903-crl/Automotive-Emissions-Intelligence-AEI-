# Model Evaluation Report

## Metrics Explained
- **R2 Score**: Represents the proportion of variance in the dependent variable explained by the independent variables. Higher is better (max 1.0).
- **RMSE**: Root Mean Squared Error. Represents the average prediction error in the same units as the target variable. Lower is better, heavily penalizes large errors.
- **MAE**: Mean Absolute Error. The average absolute difference between prediction and actual. Lower is better.

### Linear Regression
- **R2 Score**: 0.9919
- **RMSE**: 4.7993
- **MAE**: 3.3754

### Random Forest Regressor
- **R2 Score**: 0.9974
- **RMSE**: 2.7118
- **MAE**: 1.8063

### XGBoost Regressor
- **R2 Score**: 0.9978
- **RMSE**: 2.5154
- **MAE**: 1.7564

## Conclusion
The best performing model is **XGBoost Regressor** with an R2 score of 0.9978. This model was chosen because it achieved the highest explanatory power (R2) while minimizing average errors (RMSE/MAE). Tree-based models (like Random Forest and XGBoost) typically capture non-linear interactions better than simple Linear Regression.
