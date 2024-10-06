import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Path to your CSV (
path = "path to your csv"
df = pd.read_csv(path)
df.rename(columns={'time_abs(%Y-%m-%dT%H:%M:%S.%f)': 'date', 'time_rel(sec)': 'time', 'velocity(m/s)': 'velocity'}, inplace=True)

# Interpolate missing data
df.interpolate(method='linear', inplace=True)

# --- Handle outliers (clip velocity values) ---
# Clip values that are outside a reasonable range (modify thresholds based on your data)
df['velocity'] = np.clip(df['velocity'], df['velocity'].quantile(0.01), df['velocity'].quantile(0.99))

# Normalize the data
scaler = MinMaxScaler()
df[['velocity']] = scaler.fit_transform(df[['velocity']])

# --- Smoothing the signal before feature extraction ---
df['smoothed_velocity'] = df['velocity'].rolling(window=50, min_periods=1).mean()

def extract_features(df, window_size=100):
    windows = []
    for start in range(0, len(df), window_size):
        window = df['smoothed_velocity'][start:start+window_size]
        if len(window) == window_size:
            windows.append([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                np.median(window)
            ])
    return np.array(windows)

# Extract features using smoothed velocity
features = extract_features(df)

# --- Isolation Forest for anomaly detection ---
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(features)

# Predict anomalies (1 = normal, -1 = anomaly)
anomalies_if = iso_forest.predict(features)

# Identify anomaly indices
anomaly_indices_if = np.where(anomalies_if == -1)[0]

# --- Autoencoder for anomaly detection ---
autoencoder = Sequential([
    Dense(32, activation='relu', input_shape=(features.shape[1],)),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(features.shape[1], activation='sigmoid')
])

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(features, features, epochs=50, batch_size=32, shuffle=True)

# Reconstruction error
reconstructions = autoencoder.predict(features)
mse = np.mean(np.power(features - reconstructions, 2), axis=1)

# --- Adaptive thresholding based on reconstruction error ---
threshold = np.percentile(mse, 95)  # Adjust threshold (95th percentile)
anomalies_ae = mse > threshold
anomaly_indices_ae = np.where(anomalies_ae)[0]

# --- Visualization of anomalies ---
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['velocity'], label='Seismic Signal', alpha=0.2)

# Mark anomalies from Isolation Forest in red
plt.scatter(df['time'][anomaly_indices_if * 100], df['velocity'][anomaly_indices_if * 100], 
            color='red', label='Isolation Forest Anomalies', alpha=0.5)

# Mark anomalies from Autoencoder in green
plt.scatter(df['time'][anomaly_indices_ae * 100], df['velocity'][anomaly_indices_ae * 100], 
            color='blue', label='Autoencoder Anomalies', alpha=0.5)

plt.xlabel('Time')
plt.ylabel('Velocity')
#title ='Anomalies Detected in Seismic Data: ', str(i)
plt.title('Anomalies Detected in Seismic Data: ' + str(i))
plt.legend()
plt.show()

# Plot the smoothed signal for identifying outliers or spikes
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['smoothed_velocity'])
plt.xlabel('Time')
plt.ylabel('Smoothed Velocity')
plt.title('Smoothed Seismic Signal: ' + str(i))
plt.show()