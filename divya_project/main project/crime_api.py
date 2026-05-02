import os
import threading
import gradio as gr
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from pyngrok import ngrok, conf

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
CITIES   = ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Ahmedabad', 'Pune']
TIMES    = ['Morning', 'Afternoon', 'Evening', 'Night', 'Midnight']
WEATHERS = ['sunny', 'cloudy', 'rainy', 'foggy', 'stormy']
FEATURES = ['city', 'time', 'weather', 'cctv', 'police', 'traffic', 'speed', 'temp', 'rainfall', 'accidents', 'complaints']

# ─── LABEL ENCODERS ───────────────────────────────────────────────────────────
le_city    = LabelEncoder().fit(CITIES)
le_time    = LabelEncoder().fit(TIMES)
le_weather = LabelEncoder().fit(WEATHERS)

# ─── LOAD DATASET ─────────────────────────────────────────────────────────────
def load_data():
    csv_path = 'crime_dataset.csv'

    if os.path.exists(csv_path):
        print("Loading dataset from crime_dataset.csv ...")
        df = pd.read_csv(csv_path)

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
        print("Dataset loaded: " + str(len(df)) + " rows")
        return df

    else:
        print("crime_dataset.csv not found — using synthetic sample data")
        np.random.seed(42)
        n = 1000

        df = pd.DataFrame({
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

        return df

# ─── ENCODE DATAFRAME ─────────────────────────────────────────────────────────
def encode_df(df):
    d = df.copy()
    d['city']    = le_city.transform(d['city'])
    d['time']    = le_time.transform(d['time'])
    d['weather'] = le_weather.transform(d['weather'])
    return d

# ─── TRAIN MODEL ──────────────────────────────────────────────────────────────
data  = load_data()
X     = encode_df(data[FEATURES])
y     = data['severity']
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)
print("Model trained successfully")

# ─── PREDICT FUNCTION ─────────────────────────────────────────────────────────
def predict(city, time, weather, cctv, police, traffic, speed, temp, rainfall, accidents, complaints):
    try:
        row = pd.DataFrame([{
            'city':       city,
            'time':       time,
            'weather':    weather.lower(),
            'cctv':       int(cctv),
            'police':     int(police),
            'traffic':    int(traffic),
            'speed':      int(speed),
            'temp':       int(temp),
            'rainfall':   int(rainfall),
            'accidents':  int(accidents),
            'complaints': int(complaints)
        }])

        row_enc = encode_df(row)
        score   = float(np.clip(model.predict(row_enc)[0], 1, 10))
        score   = round(score, 3)

        if score >= 8:
            risk = 'CRITICAL'
        elif score >= 6:
            risk = 'HIGH'
        elif score >= 4:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        return score, risk

    except Exception as e:
        return 0.0, 'Error: ' + str(e)

# ─── FLASK APP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'super_secret_crime_key_for_demo'
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error=True)
    return render_template('login.html', error=False)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/logout', methods=['GET'])
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'model': 'RandomForest',
        'rows_trained': len(data)
    })

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    file_path = 'crime_dataset.csv'
    file.save(file_path)
    
    global data, X, y, model
    data = load_data()
    X = encode_df(data[FEATURES])
    y = data['severity']
    model.fit(X, y)
    
    # Calculate analytics
    total_incidents = len(data)
    avg_severity = round(data['severity'].mean(), 2)
    cities_count = data['city'].nunique()
    accuracy = round(model.score(X, y) * 100, 1)
    
    city_sev = data.groupby('city')['severity'].mean().round(2).to_dict()
    time_sev = data.groupby('time')['severity'].mean().round(2).to_dict()
    
    # Calculate crime type distribution
    if 'crime_type' in data.columns:
        crime_type_counts = data['crime_type'].value_counts()
        crime_types = crime_type_counts.head(6).to_dict()
    elif 'crimeType' in data.columns:
        crime_type_counts = data['crimeType'].value_counts()
        crime_types = crime_type_counts.head(6).to_dict()
    else:
        crime_types = {'Theft': 35, 'Robbery': 22, 'Assault': 18, 'Fraud': 12, 'Vandalism': 8, 'Other': 5}
    
    importances = dict(zip(FEATURES, model.feature_importances_))
    importances = {k: round(v * 100, 1) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

    # Deep city profiles for map popups
    city_details = {}
    for city, grp in data.groupby('city'):
        avg_s = round(grp['severity'].mean(), 2)
        count = len(grp)
        if 'crime_type' in grp.columns:
            top_crime = grp['crime_type'].mode()[0] if not grp['crime_type'].mode().empty else 'N/A'
        elif 'crimeType' in grp.columns:
            top_crime = grp['crimeType'].mode()[0] if not grp['crimeType'].mode().empty else 'N/A'
        else:
            top_crime = 'Theft'
        max_s = round(grp['severity'].max(), 2)
        min_s = round(grp['severity'].min(), 2)
        risk = 'CRITICAL' if avg_s >= 7.5 else 'HIGH' if avg_s >= 6.0 else 'MEDIUM' if avg_s >= 4.5 else 'LOW'
        city_details[city] = {
            'count': count,
            'avg_severity': avg_s,
            'max_severity': max_s,
            'min_severity': min_s,
            'top_crime': top_crime,
            'risk': risk
        }
    
    return jsonify({
        'status': 'success',
        'message': 'Dataset uploaded and model retrained.',
        'analytics': {
            'total_incidents': total_incidents,
            'avg_severity': avg_severity,
            'cities_count': cities_count,
            'accuracy': accuracy,
            'city_sev': city_sev,
            'time_sev': time_sev,
            'crime_types': crime_types,
            'importances': importances,
            'city_details': city_details
        }
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def api_predict():
    if request.method == 'OPTIONS':
        res = jsonify({})
        res.headers['Access-Control-Allow-Origin']  = '*'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        res.headers['Access-Control-Allow-Methods'] = 'POST'
        return res

    d = request.get_json()

    score, risk = predict(
        city       = d.get('city', 'Delhi'),
        time       = d.get('time', 'Morning'),
        weather    = d.get('weather', 'sunny'),
        cctv       = d.get('cctv', 2),
        police     = d.get('police', 1),
        traffic    = d.get('traffic', 2),
        speed      = d.get('speed', 25),
        temp       = d.get('temp', 32),
        rainfall   = d.get('rainfall', 0),
        accidents  = d.get('accidents', 2),
        complaints = d.get('complaints', 2)
    )

    return jsonify({
        'predicted_severity': score,
        'risk_level': risk,
        'status': 'success'
    })

# ─── GRADIO UI ────────────────────────────────────────────────────────────────
gradio_app = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=CITIES,   value='Delhi',   label='City'),
        gr.Dropdown(choices=TIMES,    value='Morning', label='Time of Day'),
        gr.Dropdown(choices=WEATHERS, value='sunny',   label='Weather Condition'),
        gr.Number(value=2,  label='Nearby CCTV Cameras'),
        gr.Number(value=1,  label='Police Stations Nearby'),
        gr.Number(value=2,  label='Traffic Congestion Level (1-10)'),
        gr.Number(value=25, label='Traffic Avg Speed (KMH)'),
        gr.Number(value=32, label='Temperature (C)'),
        gr.Number(value=0,  label='Rainfall (mm)'),
        gr.Number(value=2,  label='Road Accidents Reported'),
        gr.Number(value=2,  label='Public Complaints')
    ],
    outputs=[
        gr.Number(label='Predicted Crime Severity (1-10)'),
        gr.Text(label='Risk Level')
    ],
    title='Crime Severity Prediction',
    description='Crime Pattern Detection using Geospatial and Temporal Analysis'
)

# ─── LAUNCH ───────────────────────────────────────────────────────────────────
def run_flask():
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)

if __name__ == '__main__':

    # If ngrok.exe is in the same folder use it directly
    local_ngrok = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ngrok.exe')
    if os.path.exists(local_ngrok):
        conf.get_default().ngrok_path = local_ngrok
        print("Using local ngrok.exe")

    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print("Flask API running at http://localhost:7860")

    # Start ngrok tunnel
    try:
        public_url = ngrok.connect(7860)
        print("")
        print("PUBLIC URL  : " + str(public_url))
        print("API ENDPOINT: " + str(public_url) + "/predict")
        print("")
        print("Paste the API ENDPOINT above into your Netlify dashboard")
        print("")
    except Exception as e:
        print("ngrok error: " + str(e))
        print("Running without ngrok. Use http://localhost:7860/predict for local testing.")

    # Start Gradio UI
    gradio_app.launch(
        server_name='0.0.0.0',
        server_port=7861,
        share=True
    )
