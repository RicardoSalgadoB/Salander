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
##Todos los paths pueden ser omitidos porque son dll para librerias compiladas con cuda
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

def procesar_mseed(mseed_file):
    tiempos = []
    try:
        stream = read(mseed_file)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {mseed_file}.")
        return
    except Exception as e:
        print(f"Error al procesar el archivo {mseed_file}: {e}.")
        return

    minfreq = 0.5
    maxfreq = 1.0
    stream_filt = stream.copy()
    stream_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
    trace_filt = stream_filt.traces[0].copy()
    trace_times_filt = trace_filt.times()
    trace_data_filt = trace_filt.data

    f, t, sxx = signal.spectrogram(trace_data_filt, trace_filt.stats.sampling_rate)

    trace_data_filt = wiener(trace_data_filt, mysize=None, noise=None)

    fig = plt.figure(figsize=(10, 10))
    ax2 = plt.subplot(2, 1, 1)
    vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    ax2.set_xlim([min(trace_times_filt), max(trace_times_filt)])
    ax2.set_xlabel(f'(S)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    cbar = plt.colorbar(vals, orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

    output_path = f"spectrogram_row_0.png"
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Espectrograma guardado en: {output_path}")

    model = YOLO('best.pt')  

    spectrogram_image = cv2.imread(output_path)
    if spectrogram_image is None:
        print(f"Error al cargar el espectrograma: {output_path}")
    else:
        results = model(spectrogram_image)
        for result in results:
            boxes = result.boxes  
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy().astype(int)

                x_start = 125  
                x_end = 900    

                spectrogram_active_width = x_end - x_start
                time_range = t[-1] - t[0]

                # Convertir las coordenadas del bounding box a tiempo relativo
                time_at_xmin = t[0] + ((xmin - x_start) / spectrogram_active_width) * time_range
                time_at_xmax = t[0] + ((xmax - x_start) / spectrogram_active_width) * time_range
                time_offset = 50

                # Asegurarse de que xmin y xmax estén dentro del rango del espectrograma
                if xmin < x_start or xmax > x_end:
                    print(f"Advertencia: Bounding box fuera del área activa del espectrograma. xmin={xmin}, xmax={xmax}")

                # Imprimir los tiempos corregidos
                print(f"Detección en el espectrograma entre t={time_at_xmin:.2f}s y t={time_at_xmax:.2f}s")
                tiempos.append(time_at_xmin)
                indices_deteccion = np.where((trace_times_filt >= time_at_xmin-time_offset) & (trace_times_filt <= time_at_xmax+time_offset))[0]

                # Si hay datos dentro de los tiempos de detección
                if len(indices_deteccion) > 0:
                    # Extraer los tiempos y magnitudes de velocidad dentro del rango de detección
                    tiempos_deteccion = trace_times_filt[indices_deteccion]
                    velocidad_deteccion = trace_data_filt[indices_deteccion]

                    # Crear gráfica de velocidad para cada detección
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(tiempos_deteccion, velocidad_deteccion, label='Velocidad', color='blue')

                    # Añadir una línea vertical para marcar el tiempo de llegada relativo (arrival time)
                    ax.axvline(x=time_at_xmin-time_offset, color='green', linestyle='-', label='Tiempo de llegada modelado')
                    # Configurar etiquetas y leyenda
                    ax.set_xlabel('Tiempo (s)')
                    ax.set_ylabel('Velocidad (m/s)')
                    ax.set_title(f'Magnitudes de velocidad en tiempos de detección {box.cls[0]}')
                    ax.legend()

                    # Guardar el gráfico por cada detección
                    plt.savefig(f"velocity_detection_{int(box.cls[0])}.png")
                    plt.close(fig)

                else:
                    print(f"No hay datos de velocidad en el rango entre t={time_at_xmin:.2f}s y t={time_at_xmax:.2f}s")

                # Dibujar la caja en la imagen del espectrograma
                pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], np.int32)
                pts = pts.reshape((-1, 1, 2))  # Reconfigurar para OpenCV polylines
                cv2.polylines(spectrogram_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return tiempos

# Llamar al método con el archivo .mseed
mseed_file = r"D:\Space Aps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection\data\lunar\training\data\S12_GradeA\xa.s12.00.mhz.1972-07-17HR00_evid00068.mseed"
tiempos = procesar_mseed(mseed_file)