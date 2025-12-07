# Comparison of ML models

The models use were:
- Logistic Regression
- XGBoost

### Which features were most predictive?

XGBoost:
- rsi
- return_lag_3
- macd
- return_lag_5

## Which model performed best and why?
XGBoost typically outperforms Logistic Regression

- Non-linear relationships: captures feature interactions
- Feature importance: automatically selects relevant features
- Robust to outliers: tree-based splits handle extreme values

Logistic Regression is simpler and interpretable but assumes linear relationships, which limits it for market dynamics.

## Limitations of ML in financial forecasting
- Non-stationarity: Market regimes change. Models trained on past data may not generalize
- Data leakage: Future information can leak into features
- Overfitting: Complex models can memorize noise rather than learn patterns