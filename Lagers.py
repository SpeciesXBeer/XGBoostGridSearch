!pip install xgboost

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Load the data sets (pre)
train_df = pd.read_csv('.../Training Sets/Lager Training Data - Training Data.csv')
test_df = pd.read_csv('.../Testing Sets/Lager Testing Data - Testing Data Recipe Generator.csv')

#Transform data
le = LabelEncoder()
train_df['Beer Name'] = le.fit_transform(train_df['Beer Name'])

#assign x and y
y_train = train_df['Untappd Score']
train_df.pop('Untappd Score')
test_df.pop('Untappd Score')
scaler = StandardScaler()
scaled_train_df = scaler.fit_transform(train_df)
scaled_test_df = scaler.transform(test_df)

# Create and configure the XGBoost regressor
xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

# Define the parameter grid for tuning
param_grid = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.5, 0.6, 0.7],
    'colsample_bytree': [0.5, 0.6, 0.7],
    'learning_rate': [0.01, 0.05, 0.1],
}

# Set up GridSearchCV with the XGBoost regressor and the parameter grid
grid_search = GridSearchCV(estimator=xgb_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)

# Fit the grid search to the training data
grid_search.fit(scaled_train_df, y_train)

# Print the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Train the model with the best parameters
best_xgb_regressor = grid_search.best_estimator_
best_xgb_regressor.fit(train_df, y_train)

# Make predictions on the test set
y_pred = best_xgb_regressor.predict(test_df)

#Implant New Untappd Reviews to test data frame
test_df['Untappd Score'] = y_pred

#Export Final CSV into Colab Testing Folder - Change to name you want
test_df.to_csv('.../Lager XGBoost Results scaled 3 4_6_23.csv',index=False)
