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

    # Los nombres de las columnas, siguiendo el formato de evidencia152
    time_col = 'rel_time(sec)'  # Nombre de la columna de tiempo
    velocity_col = 'velocity(c/s)'  # Nombre de la columna de velocidad

    # Extraer columnas relevantes
    rel_time = evidence_data[time_col].to_numpy()
    velocity = evidence_data[velocity_col].to_numpy()

    # Mejorar la detección de picos
    # Ajustar el umbral de altura y la distancia mínima entre picos
    height_threshold = np.mean(velocity) + 1.5 * np.std(velocity)  # Umbral de altura
    distance_threshold = 50  # Distancia mínima entre picos (ajusta según sea necesario)

    # Encontrar picos en la señal
    peaks, _ = find_peaks(velocity, height=height_threshold, distance=distance_threshold)
    
    if len(peaks) > 0:
        # Obtener el tiempo del pico más alto (inicio del sismo)
        event_start_time = rel_time[peaks[np.argmax(velocity[peaks])]]  # Hora del sismo
        
        # Para el final del sismo, considerar el último pico significativo
        event_end_time = rel_time[peaks[-1]]  # Último pico significativo (fin del sismo)
    else:
        print("No se encontraron picos en la señal.")
        return

    # Generar gráficos
    plt.figure(figsize=(12, 6))

    # Graficar la señal de velocidad
    plt.plot(rel_time, velocity, label='Velocity', color='blue')
    
    # Marcar el tiempo del evento sísmico en la gráfica
    plt.axvline(x=event_start_time, color='red', linestyle='--', label='Inicio del sismo')
    plt.text(event_start_time, max(velocity), f'Inicio sismo en t={event_start_time:.2f}s', color='red', fontsize=10)
    
    # Marcar el final del sismo en la gráfica
    plt.axvline(x=event_end_time, color='green', linestyle='--', label='Fin del sismo')
    plt.text(event_end_time, max(velocity) - 0.2e-9, f'Fin sismo en t={event_end_time:.2f}s', color='green', fontsize=10)

    plt.title(f'Seismic Velocity Over Time: {csv_file}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (c/s)')
    plt.legend()
    plt.grid()

    # Guardar la figura
    output_filename = os.path.splitext(csv_file)[0] + "_output.png"
    plt.savefig(output_filename)

    # Mostrar la figura
    plt.show()

    # Imprimir el tiempo del inicio y final del sismo detectado
    print(f'Inicio del sismo detectado en t={event_start_time:.2f} segundos.')
    print(f'Fin del sismo detectado en t={event_end_time:.2f} segundos.')

# Procesar el archivo "evidencia198.csv"
csv_file = "XB.ELYSE.02.BHV.2022-01-02HR04_evid0006.csv"  # El archivo que quieres procesar

# Procesar el archivo y generar la gráfica
process_seismic_file(csv_file)
