import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Sample dataset (you can replace this with your real data)
data = pd.DataFrame({
    'years_at_company': [1, 2, 3, 4, 5, 6, 7, 8],
    'satisfaction_level': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'average_monthly_hours': [140, 150, 160, 170, 180, 190, 200, 210],
    'salary': [25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000]
})

# Features and target
X = data[['years_at_company', 'satisfaction_level', 'average_monthly_hours']]
y = data['salary']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model, "model.pkl")

print(" scaler.pkl and model.pkl saved successfully!")
