# Importar las librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import os
from scipy.signal import wiener
from scipy import signal
from matplotlib import cm

# Obtener los datos del archivo CSV
cat_file = './data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
cat_file = os.path.join(r"D:\Space Aps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection", cat_file)
cat = pd.read_csv(cat_file)

# Iterar sobre todas las filas del archivo CSV (comenzando en la fila 1)
for row in range(1, len(cat)):
    filename = cat['filename'].iloc[row]
    print(f"Procesando archivo: {filename}")

    # Obtener el archivo .mseed correspondiente
    file = f'./data/lunar/training/data/S12_GradeA/{filename}.mseed'
    file = os.path.join(r"D:\Space Aps\space_apps_2024_seismic_detection\space_apps_2024_seismic_detection", file)

    try:
        # Intentar leer el archivo
        stream = read(file)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {filename}. Siguiente archivo...")
        continue  # Saltar al siguiente archivo
    except Exception as e:
        print(f"Error al procesar el archivo {filename}: {e}. Siguiente archivo...")
        continue  # Saltar al siguiente archivo en caso de otro tipo de error

    # Si el archivo fue leído correctamente, continuar con el procesamiento
    sampling_rate = stream[0].stats.sampling_rate
    start_time = stream[0].stats.starttime
    duration = stream[0].stats.endtime - stream[0].stats.starttime

    # Obtener el tiempo de llegada relativo
    arrival_time_relative = cat['time_rel(sec)'].iloc[row]
    arrival_time = start_time + pd.to_timedelta(arrival_time_relative, unit='s')
    arrival = arrival_time_relative

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

    # Parámetros STA/LTA
    df = trace_filt.stats.sampling_rate
    sta_len = 140
    lta_len = 800
    cft = classic_sta_lta(trace_data_filt, int(sta_len * df), int(lta_len * df))

    # Definir los umbrales para las detecciones
    thr_on = 3
    thr_off = 0.5
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))

    # Crear la gráfica del espectrograma y la señal
    fig = plt.figure(figsize=(10, 10))

    # Graficar el espectrograma
    ax2 = plt.subplot(2, 1, 1)
    vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
    ax2.set_xlim([min(trace_times_filt), max(trace_times_filt)])
    ax2.set_xlabel(f'(S)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    #ax2.axvline(x=arrival, c='red') ##Not graphed for training
    cbar = plt.colorbar(vals, orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

    # Guardar la gráfica como una imagen .png
    output_path = f"spectrogram_row_{row}.png"
    plt.savefig(output_path)
    print(f"Espectrograma guardado en: {output_path}")

    # Cerrar la figura para evitar sobrecargar la memoria
    plt.close(fig)
