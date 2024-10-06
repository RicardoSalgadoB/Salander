import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

# Función para cargar y graficar la señal sísmica desde un archivo CSV
def process_seismic_file(csv_file):
    # Cargar los datos de la señal sísmica desde el archivo CSV
    evidence_data = pd.read_csv(csv_file)

    # Mostrar las columnas disponibles
    print("Columnas disponibles:", evidence_data.columns.tolist())

    # Los nombres de las columnas
    time_col = 'rel_time(sec)'  # Nombre de la columna de tiempo
    velocity_col = 'velocity(c/s)'  # Nombre de la columna de velocidad

    # Extraer columnas relevantes
    time_rel = evidence_data[time_col].to_numpy()
    velocity = evidence_data[velocity_col].to_numpy()

    # Mejorar la detección de picos
    height_threshold = np.mean(velocity) + 2 * np.std(velocity)  # Umbral de altura ajustado
    distance_threshold = 50  # Distancia mínima entre picos (ajusta según sea necesario)

    # Encontrar picos en la señal
    peaks, _ = find_peaks(velocity, height=height_threshold, distance=distance_threshold)

    if len(peaks) > 0:
        # Obtener el tiempo del pico más alto
        event_time = time_rel[peaks[np.argmax(velocity[peaks])]]  # Hora del sismo
    else:
        print("No se encontraron picos en la señal.")
        return

    # Generar gráficos
    plt.figure(figsize=(12, 6))

    # Graficar la señal de velocidad
    plt.plot(time_rel, velocity, label='Velocity', color='blue')
    
    # Marcar el tiempo del evento sísmico en la gráfica
    plt.axvline(x=event_time, color='red', linestyle='--', label='Sismo detectado')
    plt.text(event_time, max(velocity), f'Sismo en t={event_time:.2f}s', color='red', fontsize=10)

    plt.title('Seismic Velocity Over Time: xa.s12.00.mhz.1970-02-07HR00_evid00014.csv')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid()

    # Guardar la figura
    output_filename = os.path.splitext(csv_file)[0] + "_output.png"
    plt.savefig(output_filename)

    # Mostrar la figura
    plt.show()

    # Imprimir el tiempo del sismo detectado
    print(f'Sismo detectado en t={event_time:.2f} segundos.')

# Procesar el archivo "xa.s12.00.mhz.1970-02-07HR00_evid00014.csv"
csv_file = "/Users/r.salgadob./Desktop/Salander/space_apps_2024_seismic_detection/data/mars/test/data/XB.ELYSE.02.BHV.2021-10-11HR23_evid0011.csv"  # El archivo que quieres procesar

# Procesar el archivo y generar la gráfica
process_seismic_file(csv_file)
