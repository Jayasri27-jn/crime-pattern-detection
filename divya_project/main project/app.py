import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import folium

CITY_COORDS = {
    'Delhi': [28.7041, 77.1025],
    'Mumbai': [19.0760, 72.8777],
    'Bangalore': [12.9716, 77.5946],
    'Chennai': [13.0827, 80.2707],
    'Kolkata': [22.5726, 88.3639],
    'Hyderabad': [17.3850, 78.4867],
    'Ahmedabad': [23.0225, 72.5714],
    'Pune': [18.5204, 73.8567]
}

CITIES   = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Ahmedabad', 'Pune']
TIMES    = ['Morning', 'Afternoon', 'Evening', 'Night', 'Midnight']
WEATHERS = ['sunny', 'cloudy', 'rainy', 'foggy', 'stormy']
CRIME_TYPES = ['Theft', 'Robbery', 'Assault', 'Fraud', 'Vandalism', 'Burglary', 'Carjacking', 'Pickpocketing']

le_city    = LabelEncoder().fit(CITIES)
le_time    = LabelEncoder().fit(TIMES)
le_weather = LabelEncoder().fit(WEATHERS)

FEATURES = ['city', 'time', 'weather', 'cctv', 'police', 'traffic',
            'speed', 'temp', 'rainfall', 'accidents', 'complaints']

def load_data():
    if os.path.exists('crime_dataset.csv'):
        df = pd.read_csv('crime_dataset.csv')
        rename_map = {
            'City': 'city',
            'Time_of_Day': 'time',
            'Weather_Condition': 'weather',
            'Nearby_CCTV_Cameras': 'cctv',
            'Police_Stations_Nearby': 'police',
            'Traffic_Congestion': 'traffic',
            'Traffic_Avg_Speed': 'speed',
            'Temperature': 'temp',
            'Rainfall': 'rainfall',
            'Road_Accidents': 'accidents',
            'Public_Complaints': 'complaints',
            'Crime_Severity': 'severity'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df
    else:
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            'city':       np.random.choice(CITIES, n),
            'time':       np.random.choice(TIMES, n),
            'weather':    np.random.choice(WEATHERS, n),
            'cctv':       np.random.randint(0, 10, n),
            'police':     np.random.randint(0, 5, n),
            'traffic':    np.random.randint(1, 10, n),
            'speed':      np.random.randint(5, 80, n),
            'temp':       np.random.randint(15, 45, n),
            'rainfall':   np.random.randint(0, 300, n),
            'accidents':  np.random.randint(0, 10, n),
            'complaints': np.random.randint(0, 10, n),
            'severity':   np.random.uniform(1, 10, n)
        })

def encode_df(df):
    d = df.copy()
    d['city']    = le_city.transform(d['city'])
    d['time']    = le_time.transform(d['time'])
    d['weather'] = le_weather.transform(d['weather'])
    return d

data  = load_data()
X     = encode_df(data[FEATURES])
y     = data['severity']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

def predict(city, location, time, crime_type, cctv, police, traffic,
            speed, weather, temp, rainfall, accidents, complaints):
    try:
        row = pd.DataFrame([{
            'city':       city,
            'time':       time,
            'weather':    weather.lower() if weather else 'sunny',
            'cctv':       int(cctv),
            'police':     int(police),
            'traffic':    int(traffic),
            'speed':      int(speed),
            'temp':       int(temp),
            'rainfall':   int(rainfall),
            'accidents':  int(accidents),
            'complaints': int(complaints)
        }])
        score = float(np.clip(model.predict(encode_df(row))[0], 1, 10))
        score = round(score, 3)
        risk  = 'CRITICAL' if score >= 8 else 'HIGH' if score >= 6 else 'MEDIUM' if score >= 4 else 'LOW'
        return score
    except Exception as e:
        return 0.0

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        
        city = data.get('city', 'Delhi')
        location = data.get('location', 'Unknown')
        time = data.get('time', 'Morning')
        crime_type = data.get('crimeType', 'theft')
        cctv = int(data.get('cctv', 0))
        police = int(data.get('police', 0))
        traffic = int(data.get('traffic', 0))
        speed = int(data.get('speed', 0))
        weather = data.get('weather', 'sunny')
        temp = int(data.get('temp', 0))
        rainfall = int(data.get('rainfall', 0))
        accidents = int(data.get('accidents', 0))
        complaints = int(data.get('complaints', 0))
        
        score = predict(city, location, time, crime_type, cctv, police, traffic,
                        speed, weather, temp, rainfall, accidents, complaints)
        
        risk = 'CRITICAL' if score >= 8 else 'HIGH' if score >= 6 else 'MEDIUM' if score >= 4 else 'LOW'
        
        return jsonify({
            'status': 'success',
            'predicted_severity': score,
            'risk_level': risk
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
