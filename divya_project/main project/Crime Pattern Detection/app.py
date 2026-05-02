import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Crime Analysis Dashboard", layout="wide")

st.title("🚔 Crime Pattern Detection & Analysis")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    import os
    if not os.path.exists("clean_crime_dataset.csv"):
        st.error("Dataset 'clean_crime_dataset.csv' not found. Please check the file path.")
        st.stop()
    df = pd.read_csv("clean_crime_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    # Ensure Time is handled correctly before extracting hour
    df['Hour'] = pd.to_datetime(df['Time'], errors='coerce').dt.hour
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

df = load_data()

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔍 Filters")

# Add a download button for the filtered data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

selected_area = st.sidebar.multiselect("Select Area", df['Area'].unique(), default=df['Area'].unique())
selected_crime = st.sidebar.multiselect("Select Crime Type", df['Crime_Type'].unique(), default=df['Crime_Type'].unique())

filtered_data = df[(df['Area'].isin(selected_area)) & (df['Crime_Type'].isin(selected_crime))]

if filtered_data.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

csv = convert_df(filtered_data)
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name='filtered_crime_data.csv',
    mime='text/csv',
)

# -------------------------------
# KPI METRICS
# -------------------------------
st.subheader("📊 Key Insights")

col1, col2, col3 = st.columns(3)

col1.metric("Total Crimes", len(filtered_data))
col2.metric("Areas Covered", filtered_data['Area'].nunique())
col3.metric("Crime Types", filtered_data['Crime_Type'].nunique())

# -------------------------------
# MONTHLY TREND
# -------------------------------
st.subheader("📈 Monthly Crime Trend")

# Group by year and month, then create a period index for better plotting
monthly = filtered_data.groupby(['Year', 'Month']).size().reset_index(name='Crime Count')
monthly['Date_Label'] = pd.to_datetime(monthly[['Year', 'Month']].assign(Day=1))
monthly = monthly.set_index('Date_Label').sort_index()

fig, ax = plt.subplots()
ax.plot(monthly.index, monthly['Crime Count'], marker='o', linestyle='-')
ax.set_title("Crime Trend Over Time")
ax.set_ylabel("Number of Crimes")
plt.xticks(rotation=45)

st.pyplot(fig)

# -------------------------------
# HEATMAP
# -------------------------------
st.subheader("🗺️ Crime Heatmap")

# Drop rows with missing coordinates for mapping
map_data = filtered_data.dropna(subset=['Latitude', 'Longitude'])

if not map_data.empty:
    map_center = [map_data['Latitude'].mean(), map_data['Longitude'].mean()]
else:
    map_center = [0, 0]

m = folium.Map(location=map_center, zoom_start=12, tiles="cartodbpositron")

heat_data = map_data[['Latitude', 'Longitude']].values.tolist()
HeatMap(heat_data, radius=15, blur=20).add_to(m)

# -------------------------------
# DBSCAN HOTSPOTS (Visualized on Map)
# -------------------------------
if len(map_data) >= 10:
    coords = map_data[['Latitude', 'Longitude']].values
    db = DBSCAN(eps=0.005, min_samples=10).fit(coords)
    map_data = map_data.copy()
    map_data['Cluster'] = db.labels_

    # Add markers for clusters (Hotspots)
    for idx, row in map_data[map_data['Cluster'] != -1].iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            popup=f"Hotspot: {row['Crime_Type']}"
        ).add_to(m)

st_folium(m, width=700, height=500)

# -------------------------------
# MACHINE LEARNING
# -------------------------------
st.subheader("🤖 Crime Prediction Model")

@st.cache_resource
def train_model(data):
    df_ml = data.copy()
    
    # Using separate encoders for each column to maintain correct mappings
    le_crime = LabelEncoder()
    le_loc = LabelEncoder()
    le_area = LabelEncoder()
    
    df_ml['Crime_Type'] = le_crime.fit_transform(df_ml['Crime_Type'])
    df_ml['Location'] = le_loc.fit_transform(df_ml['Location'])
    df_ml['Area'] = le_area.fit_transform(df_ml['Area'])
    
    X = df_ml[['Location','Area','Hour']]
    y = df_ml['Crime_Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le_crime, le_loc, le_area, acc

# Train model on the full dataset to ensure all categories are covered
model, le_crime, le_loc, le_area, accuracy = train_model(df)

# Prediction UI
st.write("### 🔮 Predict Crime Type")
st.info(f"Model Confidence (Accuracy): {accuracy:.2%}")

col_p1, col_p2 = st.columns(2)
with col_p1:
    selected_loc_name = st.selectbox("Select Location", le_loc.classes_)
with col_p2:
    selected_area_name = st.selectbox("Select Area for Prediction", le_area.classes_)
hour = st.slider("Hour", 0, 23, 12)

if st.button("Predict"):
    # Encode inputs using the specific encoders
    loc_enc = le_loc.transform([selected_loc_name])[0]
    area_enc = le_area.transform([selected_area_name])[0]
    
    # Pass as DataFrame to match feature names used during training
    input_data = pd.DataFrame([[loc_enc, area_enc, hour]], columns=['Location', 'Area', 'Hour'])
    pred = model.predict(input_data)
    st.success(f"Predicted Crime: {le_crime.inverse_transform(pred)[0]}")

# -------------------------------
# RISK LEVEL
# -------------------------------
st.subheader("🚨 Area Risk Levels")

risk = filtered_data.groupby('Area').size().reset_index(name='Crime_Count')

def risk_label(x):
    if x > 100:
        return "High 🔴"
    elif x > 50:
        return "Medium 🟠"
    else:
        return "Low 🟢"

risk['Risk_Level'] = risk['Crime_Count'].apply(risk_label)

st.dataframe(risk)

# -------------------------------
# HOURLY CRIME
# -------------------------------
st.subheader("⏰ Crime by Hour")

hourly = filtered_data.groupby('Hour').size()

fig2, ax2 = plt.subplots()
hourly.plot(kind='bar', ax=ax2)
ax2.set_title("Crime by Hour")

st.pyplot(fig2)