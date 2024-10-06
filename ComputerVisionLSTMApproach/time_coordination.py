import os
# Correct the paths with either double backslashes or raw string literals
c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
a = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib"
b = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include"
gstreamerPath = r"C:\gstreamer\1.0\msvc_x86_64\bin"

# Adding the required directories to the DLL search path
os.add_dll_directory(gstreamerPath)
os.add_dll_directory(c)
os.add_dll_directory(a)
os.add_dll_directory(b)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import wiener
from scipy import signal
from matplotlib import cm
import cv2
from ultralytics import YOLO

# Ruta al archivo CSV
cat_file = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
cat_file = os.path.join(r"D:\Space Aps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection", cat_file)
cat = pd.read_csv(cat_file)

# Ruta del archivo que quieres procesar
row = 5  # Puedes cambiar la fila a procesar
filename = cat['filename'].iloc[row]
print(f"Procesando archivo: {filename}")

# Obtener el archivo .mseed correspondiente
file = f'./data/lunar/training/data/S12_GradeA/{filename}.mseed'
file = os.path.join(r"D:\Space Aps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection", file)

try:   
    # Intentar leer el archivo
    stream = read(file)
except FileNotFoundError:
    print(f"Archivo no encontrado: {filename}.")
except Exception as e:
    print(f"Error al procesar el archivo {filename}: {e}.")

# Si el archivo fue leído correctamente, continuar con el procesamiento
sampling_rate = stream[0].stats.sampling_rate
start_time = stream[0].stats.starttime
duration = stream[0].stats.endtime - stream[0].stats.starttime

# Obtener el tiempo de llegada relativo
arrival_time_relative = cat['time_rel(sec)'].iloc[row]
arrival_time = start_time + pd.to_timedelta(arrival_time_relative, unit='s')
arrival = arrival_time_relative
print("Real arrival time", arrival)
# Filtrar el stream con un filtro de paso de banda
minfreq = 0.5
maxfreq = 1.0
stream_filt = stream.copy()
stream_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
trace_filt = stream_filt.traces[0].copy()
trace_times_filt = trace_filt.times()
trace_data_filt = trace_filt.data

# Calcular el espectrograma
f, t, sxx = signal.spectrogram(trace_data_filt, trace_filt.stats.sampling_rate)

# Aplicar el filtro de Wiener
trace_data_filt = wiener(trace_data_filt, mysize=None, noise=None)

# Graficar el espectrograma
fig = plt.figure(figsize=(10, 10))
ax2 = plt.subplot(2, 1, 1)
vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
ax2.set_xlim([min(trace_times_filt), max(trace_times_filt)])
ax2.set_xlabel(f'(S)', fontweight='bold')
ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
cbar = plt.colorbar(vals, orientation='horizontal')
cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

# Guardar el espectrograma como imagen
output_path = f"spectrogram_row_{row}.png"
plt.savefig(output_path)
plt.close(fig)
print(f"Espectrograma guardado en: {output_path}")

# **Paso 1: YOLOv8 para detección sobre el espectrograma**
# Cargar modelo YOLOv8
model = YOLO('best.pt')  # Cambia 'best.pt' por el modelo adecuado si tienes otro

# Leer el espectrograma guardado como imagen
spectrogram_image = cv2.imread(output_path)
if spectrogram_image is None:
    print(f"Error al cargar el espectrograma: {output_path}")
else:
    # Procesar la imagen con YOLOv8
    results = model(spectrogram_image)
    for result in results:
        boxes = result.boxes  # Obtener cajas detectadas
        for box in boxes:
            # Extraer coordenadas (formato xyxy)
            xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)
            # Convertir coordenada x a tiempo relativo en el espectrograma
            #spectrogram_width = spectrogram_image.shape[1]  # Ancho en píxeles
            #time_range = t[-1] - t[0]  # Duración total del espectrograma en segundos
            #time_at_xmin = t[0] + (xmin / spectrogram_width) * time_range
            #time_at_xmax = t[0] + (xmax / spectrogram_width) * time_range

            # Imprimir el tiempo en que ocurre la detección
            #print(f"Detección en el espectrograma entre t={time_at_xmin:.2f}s y t={time_at_xmax:.2f}s")

            # Coordenadas del espectrograma en la imagen (sin contar márgenes)
            x_start = 125  # Inicio del espectrograma (en píxeles)
            x_end = 900    # Fin del espectrograma (en píxeles)

            # Ancho del espectrograma útil (sin los márgenes)
            spectrogram_active_width = x_end - x_start

            # Duración total del espectrograma en segundos (t[-1] es el tiempo final, t[0] es el tiempo inicial)
            time_range = t[-1] - t[0]

            # Convertir las coordenadas del bounding box a tiempo relativo
            time_at_xmin = t[0] + ((xmin - x_start) / spectrogram_active_width) * time_range
            time_at_xmax = t[0] + ((xmax - x_start) / spectrogram_active_width) * time_range
            time_offset = 450
            # Asegurarse de que xmin y xmax estén dentro del rango del espectrograma
            if xmin < x_start or xmax > x_end:
                print(f"Advertencia: Bounding box fuera del área activa del espectrograma. xmin={xmin}, xmax={xmax}")

            # Imprimir los tiempos corregidos
            print(f"Detección en el espectrograma entre t={time_at_xmin:.2f}s y t={time_at_xmax:.2f}s")

            indices_deteccion = np.where((trace_times_filt >= time_at_xmin-time_offset) & (trace_times_filt <= time_at_xmax+time_offset))[0]

            # Si hay datos dentro de los tiempos de detección
            if len(indices_deteccion) > 0:
                # Extraer los tiempos y magnitudes de velocidad dentro del rango de detección
                tiempos_deteccion = trace_times_filt[indices_deteccion]
                velocidad_deteccion = trace_data_filt[indices_deteccion]
                
                # Crear gráfica de velocidad
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(tiempos_deteccion, velocidad_deteccion, label='Velocidad', color='blue')
                
                # Añadir una línea vertical para marcar el tiempo de llegada relativo (arrival time)
                ax.axvline(x=arrival, color='red', linestyle='--', label='Tiempo de llegada Real')
                ax.axvline(x=time_at_xmin-time_offset, color='green', linestyle='-', label='Tiempo de llegada modelado')

                # Configurar etiquetas y leyenda
                ax.set_xlabel('Tiempo (s)')
                ax.set_ylabel('Velocidad (m/s)')
                ax.set_title('Magnitudes de velocidad en tiempos de detección')
                ax.legend()

                # Mostrar el gráfico
                plt.show()
            else:
                print(f"No hay datos de velocidad en el rango entre t={time_at_xmin:.2f}s y t={time_at_xmax:.2f}s")


            # Dibujar la caja en la imagen del espectrograma
            pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], np.int32)
            pts = pts.reshape((-1, 1, 2))  # Reconfigurar para OpenCV polylines
            cv2.polylines(spectrogram_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Añadir etiqueta y puntuación de confianza
            label = f"Clase {int(box.cls[0])} {box.conf[0]:.2f}"  # Asumiendo que box.cls contiene la clase y box.conf contiene la puntuación
            cv2.putText(spectrogram_image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la imagen del espectrograma con las detecciones
    cv2.imshow("Detecciones", spectrogram_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen con las detecciones
    output_with_detections = f"spectrogram_with_detections_row_{row}.png"
    cv2.imwrite(output_with_detections, spectrogram_image)
    print(f"Espectrograma con detecciones guardado en: {output_with_detections}")

