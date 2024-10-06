import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


def extract_features(df, window_size=100):
    windows = []
    for start in range(0, len(df), window_size):
        window = df['velocity'][start:start+window_size]
        if len(window) == window_size:
            windows.append([
                np.mean(window),
                np.std(window),
                np.max(window),
                np.min(window),
                np.median(window)
            ])
    return np.array(windows)

def main():
    # path de tu csv
    path = "D:/UsX/Escritorio/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-01-19HR00_evid00002.csv"
    df = pd.read_csv(path)
    df.rename(columns={'time_abs(%Y-%m-%dT%H:%M:%S.%f)': 'date', 'time_rel(sec)': 'time', 'velocity(m/s)': 'velocity'}, inplace=True)
    df.columns

    df.interpolate(method='linear', inplace=True)

    # Normalize the data
    scaler = MinMaxScaler()
    df[['velocity']] = scaler.fit_transform(df[['velocity']])
    # Extract features
    features = extract_features(df)

    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    iso_forest.fit(features)

    # Predict anomalies (1 = normal, -1 = anomaly)
    anomalies = iso_forest.predict(features)

    # Identify anomaly indices
    anomaly_indices = np.where(anomalies == -1)[0]
    # Build an autoencoder model
    autoencoder = Sequential([
        Dense(32, activation='relu', input_shape=(features.shape[1],)),
        Dense(16, activation='relu'),
        Dense(32, activation='relu'),
        Dense(features.shape[1], activation='sigmoid')
    ])

    # Compile and train the model
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(features, features, epochs=50, batch_size=32, shuffle=True)

    # Reconstruction error
    reconstructions = autoencoder.predict(features)
    mse = np.mean(np.power(features - reconstructions, 2), axis=1)

    # Thresholding the error to identify anomalies
    threshold = np.percentile(mse, 95)  # Set threshold as the 95th percentile
    anomalies = mse > threshold
    anomaly_indices = np.where(anomalies)[0]

    # Plot time series with anomalies
    plt.plot(df['time'], df['velocity'], label='Seismic Signal')
    plt.scatter(df['time'][anomaly_indices], df['velocity'][anomaly_indices], color='red', label='Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Anomalies in Seismic Data')
    plt.legend()
    plt.show()