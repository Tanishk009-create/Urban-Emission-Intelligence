from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import datetime
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({
        "status": "Algaerithm AI Backend Running 🤖",
        "routes": [
            "GET /get_emission_risk",
            "POST /ingest_sensor_data"
        ]
    })

try:
    rf_model = joblib.load('activity_classifier.pkl')
    print("🤖 Random Forest Activity Classifier Loaded!")
except Exception as e:
    print("⚠️ Warning: activity_classifier.pkl not found. Run train_model.py first!")
    rf_model = None

sensor_data_store = []

@app.route('/ingest_sensor_data', methods=['POST'])
def ingest_data():
    data = request.json
    if not data or 'lat' not in data or 'lng' not in data or 'noise_db' not in data:
        return jsonify({"error": "Missing sensor data"}), 400
    data['timestamp'] = datetime.datetime.now().isoformat()
    sensor_data_store.append(data)
    if len(sensor_data_store) > 100:
        sensor_data_store.pop(0)
    return jsonify({"status": "success", "message": "Data ingested"}), 200

@app.route('/get_emission_risk', methods=['GET'])
def get_risk():
    lat = request.args.get('lat', default=28.6139, type=float)
    lng = request.args.get('lng', default=77.2090, type=float)
    
    # 1. Fetch Weather & AQI
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true"
        weather_response = requests.get(weather_url).json()
        wind_speed = weather_response.get('current_weather', {}).get('windspeed', 10)
        temperature = weather_response.get('current_weather', {}).get('temperature', 25)

        aqi_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lng}&current=pm10,pm2_5"
        aqi_response = requests.get(aqi_url).json()
        pm25 = aqi_response.get('current', {}).get('pm2_5', 50)
        pm10 = aqi_response.get('current', {}).get('pm10', 60)
    except:
        wind_speed, temperature, pm25, pm10 = 10, 25, 50, 60 # Fallback if API fails

    # 2. Get local noise (Check if the mobile node sent it directly, otherwise fallback)
    passed_noise = request.args.get('noise_db', type=float)
    if passed_noise is not None:
        local_noise = passed_noise
    else:
        local_noise = sensor_data_store[-1]['noise_db'] if sensor_data_store else 50

    # 3. USE THE AI MODEL TO CLASSIFY ACTIVITY
    activity_class = "Normal"
    local_penalty = 0
    
    if rf_model:
        features = np.array([[wind_speed, temperature, pm25, pm10, local_noise]])
        prediction = rf_model.predict(features)[0]
        
        if prediction == 1:
            activity_class = "Traffic Congestion Detected"
            local_penalty = 25
        elif prediction == 2:
            activity_class = "Construction Activity Detected"
            local_penalty = 45

    # 4. Calculate Final Risk Score
    base_risk = 20
    wind_factor = max(0, 20 - wind_speed) 
    aqi_factor = min(((pm25 * 0.6) + (pm10 * 0.4)) / 4, 35) 
    
    total_score = min(100, base_risk + wind_factor + aqi_factor + local_penalty)
    
    risk_level = "Low"
    if total_score > 75: risk_level = "High"
    elif total_score > 50: risk_level = "Moderate"

    return jsonify({
        "coordinates": {"lat": lat, "lng": lng},
        "risk_score": round(total_score, 1),
        "risk_level": risk_level,
        "ai_classification": activity_class,
        "factors": {
            "wind_speed_kmh": wind_speed,
            "temperature_c": temperature,
            "pm2_5": pm25,
            "pm10": pm10
        }
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Algaerithm AI Backend on http://localhost:5000")
    app.run(debug=True, port=5000)