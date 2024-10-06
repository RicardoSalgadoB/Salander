import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

def load_data(file_path, velocity_column, time_column, nrows=None):
    df = pd.read_csv(file_path, usecols=[velocity_column, time_column], nrows=nrows)
    if velocity_column not in df.columns or time_column not in df.columns:
        raise ValueError(f"Las columnas especificadas no existen en el archivo CSV.")
    return df[time_column].values, df[velocity_column].values.reshape(-1, 1)

def detect_anomalies(velocity, threshold_percentile=1, bandwidth=1.0, sample_size=10000, batch_size=1000):
    if len(velocity) > sample_size:
        indices = np.random.choice(len(velocity), sample_size, replace=False)
        velocity_sample = velocity[indices]
    else:
        velocity_sample = velocity

    scaler = StandardScaler()
    velocity_scaled = scaler.fit_transform(velocity_sample)

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(velocity_scaled)

    scores = []
    for i in tqdm(range(0, len(velocity), batch_size), desc="Calculando densidades"):
        batch = velocity[i:i+batch_size]
        batch_scaled = scaler.transform(batch)
        scores.extend(kde.score_samples(batch_scaled))

    scores = np.array(scores)
    threshold = np.percentile(scores, threshold_percentile)
    anomalies = scores <= threshold
    
    return anomalies, scores

def calculate_anomaly_density(anomalies, window_size=100):
    anomaly_density = np.convolve(anomalies, np.ones(window_size)/window_size, mode='same')
    smoothed_density = gaussian_filter1d(anomaly_density, sigma=5)
    return smoothed_density

file_path = '/Users/sebastian/Downloads/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1974-07-06HR00_evid00151.csv'
velocity_column = 'velocity(m/s)'
time_column = 'time_rel(sec)'
time, velocity = load_data(file_path, velocity_column, time_column)

print(f"Datos cargados. Forma: {velocity.shape}")

anomalies, scores = detect_anomalies(velocity, threshold_percentile=1, bandwidth=0.5, sample_size=10000, batch_size=1000)

# Calcular la densidad de anomalías
anomaly_density = calculate_anomaly_density(anomalies)

# Encontrar el punto de mayor concentración de anomalías
max_concentration_index = np.argmax(anomaly_density)

plt.figure(figsize=(15, 10))

sample_size = min(1000, len(velocity))
sample_indices = np.random.choice(len(velocity), sample_size, replace=False)
plt.scatter(time[sample_indices], velocity[sample_indices], c='b', alpha=0.3, s=1, label='Datos (muestra)')
anomaly_indices = np.where(anomalies)[0]
anomaly_sample = np.intersect1d(sample_indices, anomaly_indices)
plt.scatter(time[anomaly_sample], velocity[anomaly_sample], c='r', s=5, label='Anomalías')

# Añadir línea vertical en el punto de mayor concentración de anomalías
plt.axvline(x=time[max_concentration_index], color='green', linestyle='--', linewidth=2, 
            label='Mayor Concentración de Anomalías')

plt.title('Muestra de Datos y Anomalías Detectadas')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Velocidad (m/s)')
plt.legend()

plt.tight_layout()
plt.show()

print(f"Número total de puntos de datos: {len(velocity)}")
print(f"Número de anomalías detectadas: {np.sum(anomalies)}")
print(f"Porcentaje de anomalías: {np.mean(anomalies)*100:.2f}%")
print(f"Tiempo de mayor concentración de anomalías: {time[max_concentration_index]:.2f} segundos")

indexes_csv = pd.DataFrame({'Start_time':[time[max_concentration_index]]})
indexes_csv.to_csv('indexes_csv.csv', index=False)

df = pd.read_csv(file_path)
data_csv = pd.DataFrame()
data_csv['velocity(m/s)'] = df[velocity_column].values
data_csv['time_rel(sec)'] = df[time_column].values
data_csv.to_csv('data_csv.csv', index=False)

anomalies_csv = pd.DataFrame({'anomalies_detected': [np.sum(anomalies)],
                               'anomalies_percentage': [np.mean(anomalies)*100]})
anomalies_csv.to_csv('anomalies_csv.csv', index=False)
