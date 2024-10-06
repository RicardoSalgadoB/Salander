import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os


# Función para cargar y procesar la señal sísmica desde un archivo CSV
def process_seismic_file_and_generate_csv(csv_file):
   # Cargar los datos de la señal sísmica desde el archivo CSV
   evidence_data = pd.read_csv(csv_file)


   # Mostrar las columnas disponibles
   print("Columnas disponibles:", evidence_data.columns.tolist())


   # Los nombres de las columnas, siguiendo el formato de evidencia152
   time_col = 'time_rel(sec)'  # Nombre de la columna de tiempo
   velocity_col = 'velocity(m/s)'  # Nombre de la columna de velocidad


   # Extraer columnas relevantes
   time_rel = evidence_data[time_col].to_numpy()
   velocity = evidence_data[velocity_col].to_numpy()


   # Mejorar la detección de picos
   # Ajustar el umbral de altura y la distancia mínima entre picos
   height_threshold = np.mean(velocity) + 1.5 * np.std(velocity)  # Umbral de altura
   distance_threshold = 50  # Distancia mínima entre picos (ajusta según sea necesario)


   # Encontrar picos en la señal
   peaks, _ = find_peaks(velocity, height=height_threshold, distance=distance_threshold)
  
   if len(peaks) > 0:
       # Obtener el tiempo del pico más alto
       event_start_time = time_rel[peaks[np.argmax(velocity[peaks])]]  # Hora de inicio del sismo


       # Encontrar el tiempo final del sismo (cuando la señal vuelva a estabilizarse)
       # Aquí utilizo el pico más cercano al final de la señal que sea considerable.
       event_end_time = time_rel[peaks[-1]]  # Hora del fin del sismo (basado en el último pico relevante)
   else:
       print("No se encontraron picos en la señal.")
       return


   # Generar la gráfica de la señal sísmica
   plt.figure(figsize=(12, 6))


   # Graficar la señal de velocidad
   plt.plot(time_rel, velocity, label='Velocity', color='blue')
  
   # Marcar el tiempo del inicio y fin del evento sísmico en la gráfica
   plt.axvline(x=event_start_time, color='red', linestyle='--', label='Inicio del sismo')
   plt.axvline(x=event_end_time, color='green', linestyle='--', label='Fin del sismo')
   plt.text(event_start_time, max(velocity), f'Inicio: t={event_start_time:.2f}s', color='red', fontsize=10)
   plt.text(event_end_time, max(velocity), f'Fin: t={event_end_time:.2f}s', color='green', fontsize=10)


   plt.title(f'Seismic Velocity Over Time: {csv_file}')
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
   print(f'Inicio del sismo detectado en t={event_start_time:.2f} segundos.')
   print(f'Fin del sismo detectado en t={event_end_time:.2f} segundos.')


   # Crear un archivo CSV con los tiempos de inicio y final del sismo
   csv_output = pd.DataFrame({
       'Inicio del Sismo (s)': [event_start_time],
       'Fin del Sismo (s)': [event_end_time]
   })


   # Guardar el CSV con los tiempos del sismo
   csv_output.to_csv('grafica_en_excel.csv', index=False)


   print("Archivo 'grafica_en_excel.csv' generado con éxito.")


# Procesar el archivo "evidencia5.csv"
csv_file = "xa.s12.00.mhz.1970-01-19HR00_evid00002.csv"  # El archivo que quieres procesar


# Procesar el archivo y generar la gráfica y el archivo CSV
process_seismic_file_and_generate_csv(csv_file)
