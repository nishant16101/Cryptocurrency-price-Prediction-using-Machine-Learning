import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Load and prepare data
df = pd.read_csv('bitcoin_with metrics.csv')
df.dropna(inplace=True)
# Features and targets
features = ['Days_Since_High_Last_7_Days', '%_Diff_From_High_Last_7_Days',
            'Days_Since_Low_Last_7_Days', '%_Diff_From_Low_Last_7_Days']
target_high = '%_Diff_From_High_Next_5_Days'
target_low = '%_Diff_From_Low_Next_5_Days'

# Splitting the data
X = df[features]
y_high = df[target_high]
y_low = df[target_low]

# Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train_high, y_test_high = train_test_split(X_scaled, y_high, test_size=0.2, random_state=42)
_, _, y_train_low, y_test_low = train_test_split(X_scaled, y_low, test_size=0.2, random_state=42)

def train_model(X_train, y_train_high, y_train_low):
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestRegressor(random_state=42)
    
    # Grid Search for high target
    grid_search_high = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_high.fit(X_train, y_train_high)
    best_rf_high = grid_search_high.best_estimator_
    
    # Grid Search for low target
    grid_search_low = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_low.fit(X_train, y_train_low)
    best_rf_low = grid_search_low.best_estimator_
    
    return best_rf_high, best_rf_low  # Return trained models

def predict_outcomes(X_test, best_rf_high, best_rf_low):
    # Predictions using the trained models
    y_pred_high = best_rf_high.predict(X_test)
    y_pred_low = best_rf_low.predict(X_test)
    
    return y_pred_high, y_pred_low

# Train the models and get the best estimators
best_rf_high, best_rf_low = train_model(X_train, y_train_high, y_train_low)

# Use the best models to make predictions
y_pred_high, y_pred_low = predict_outcomes(X_test, best_rf_high, best_rf_low)

