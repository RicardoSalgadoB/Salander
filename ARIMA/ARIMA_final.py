import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import find_peaks

def detect_seismic_events_arima(data, time, order=(5,1,2), threshold=4.5, min_distance=150000):
    # Fit ARIMA model
    model = ARIMA(data, order=order)
    results = model.fit()
    
    # Make predictions
    predictions = results.predict(start=0, end=len(data)-1)
    
    # Calculate residuals
    residuals = data - predictions
    
    # Calculate standard deviation of residuals
    residual_std = np.std(residuals)
    
    # Detect anomalies
    anomalies = np.abs(residuals) > (threshold * residual_std)
    
    # Find peaks in the anomalies
    peaks, _ = find_peaks(anomalies.astype(int), distance=min_distance)
    
    return peaks, predictions, anomalies

def plot_results(time, data, peaks, predictions, anomalies):
    plt.figure(figsize=(12, 7))
    
    # Plot original data
    plt.plot(time, data, label='Original Data', alpha=0.7)
    
    # Plot ARIMA predictions
    plt.plot(time, predictions, label='ARIMA Predictions', color='green', alpha=0.5)
    
    # Plot detected events
    plt.scatter(time.iloc[peaks], data.iloc[peaks], color='red', s=100, label='Detected Events')
    
    # Plot anomalies
    plt.scatter(time[anomalies], data[anomalies], color='orange', s=20, alpha=0.5, label='Anomalies')
    
    plt.xlabel('time_rel(sec)')
    plt.ylabel('velocity(m/s)')
    plt.title('ARIMA-based Seismic Event Detection')
    plt.legend()
    plt.grid(True)
    
    # Add vertical lines for each detected event
    for peak in peaks:
        plt.axvline(x=time.iloc[peak], color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# Read the CSV file
# Assuming your CSV has columns named 'time' and 'velocity'
df = pd.read_csv('/Users/sebastian/Downloads/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1975-06-26HR00_evid00198.csv')
    
 # Ensure the data is sorted by time
df = df.sort_values('time_rel(sec)')
    
# Extract time and velocity
time = df['time_rel(sec)']  # Adjust format if needed
velocity = df['velocity(m/s)']
    
# Detect seismic events
peaks, predictions, anomalies = detect_seismic_events_arima(velocity, time)

peaks_df = pd.DataFrame({
    'time_rel(sec)': time.iloc[peaks],
    'velocity(m/s)': velocity.iloc[peaks]
})
csv_filename = 'indexes_csv.csv'
peaks_df.to_csv(csv_filename, index=False)

data_csv = pd.DataFrame()
data_csv['velocity(m/s)'] = time.values
data_csv['time_rel(sec)'] = velocity.values
data_csv.to_csv('data_csv.csv', index=False)

# Plot results
plot_results(time, velocity, peaks, predictions, anomalies)
    
# Print results
print(f"Number of seismic events detected: {len(peaks)}")
print("\nDetected events at:")
for peak in peaks:
    print(f"Time: {time.iloc[peak]}, Velocity: {velocity.iloc[peak]}")
