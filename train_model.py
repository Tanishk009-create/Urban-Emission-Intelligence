import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Generating synthetic sensor data...")
np.random.seed(42)
n_samples = 2000

# Features: wind_speed, temp, pm25, pm10, noise_db
X = np.random.rand(n_samples, 5) * [25, 45, 400, 500, 100]
y = np.zeros(n_samples)

# Logic for the AI to learn:
for i in range(n_samples):
    wind, temp, pm25, pm10, noise = X[i]
    if noise > 80 and pm10 > 250:
        y[i] = 2  
    elif noise > 70 and pm25 > 150:
        y[i] = 1 
    else:
        y[i] = 0 

print("Training Random Forest Classifier...")
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X, y)

joblib.dump(rf, 'activity_classifier.pkl')
print(" Model trained and saved as 'activity_classifier.pkl'!")