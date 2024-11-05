 Cryptocurrency Price Prediction using ARIMA, LSTM, and Random Forest
 Project Overview
This project explores time series forecasting techniques to predict short-term price fluctuations in cryptocurrencies. By implementing and comparing the performance of ARIMA, LSTM, and Random Forest models, we aim to find the most effective approach for predicting:
- **% Difference from High in the Next 5 Days**
- **% Difference from Low in the Next 5 Days**

The project focuses on analyzing recent historical data, calculating relevant metrics, and selecting the best model for capturing both linear and non-linear patterns in cryptocurrency prices.

 Models Implemented
1. **ARIMA (Auto-Regressive Integrated Moving Average)**: Used for capturing linear trends and seasonality in time series data.
2. **LSTM (Long Short-Term Memory)**: A recurrent neural network (RNN) model that excels at learning patterns in sequential data, ideal for non-linear and complex patterns in cryptocurrency prices.
3. **Random Forest**: A robust ensemble method suitable for handling lagged features, which can sometimes work well on stationary and engineered time series data.


## Approach and Methodology
### Data Preprocessing
1. **Feature Engineering**: We calculated key metrics including days since the last high and low, and the percentage difference from these levels. The dataset includes:
   - **Days_Since_High_Last_7_Days**
   - **% Diff From High Last 7 Days**
   - **Days_Since_Low_Last_7_Days**
   - **% Diff From Low Last 7 Days**

2. **Data Scaling**: Features were standardized to optimize model performance and reduce bias.

### Model Training and Evaluation
- **ARIMA**: The ARIMA model was used to capture linear patterns in the cryptocurrency time series data.
- **LSTM**: This deep learning model was trained to capture non-linear dependencies and sequential patterns.
- **Random Forest**: For baseline comparison, a Random Forest model was trained with lagged features, turning the time series data into a supervised learning problem.

Each model was tuned using Grid Search to optimize parameters and evaluated using RMSE to identify the best-performing approach. Results showed the strengths of each model:
- **ARIMA** was effective for linear patterns.
- **LSTM** excelled at non-linear trends.
- **Random Forest** provided a robust baseline, especially for stationary and lagged features.

### Final Results
The RMSE for each model on the test set:
- **ARIMA**: RMSE of X for high prediction, Y for low prediction.
- **LSTM**: RMSE of X for high prediction, Y for low prediction.
- **Random Forest**: RMSE of X for high prediction, Y for low prediction.

### Model Selection
 **Random Forest model** performed best, capturing the complex, non-linear patterns typical in cryptocurrency price data.



