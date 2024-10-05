import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import classic_sta_lta
import os

def get_data_regions(times, selected_maxima, cft, window_before=300, window_after=600):
    cft_values = cft[selected_maxima]
    total_sum = np.sum(np.abs(cft_values))
    
    regions = []
    for idx, value in zip(selected_maxima, cft_values):
        start = max(0, times[idx] - window_before)
        end = times[idx] + window_after
        regions.append((start, end, value))
    
    # Merge overlapping regions
    merged_regions = []
    for region in sorted(regions):
        if not merged_regions or merged_regions[-1][1] < region[0]:
            merged_regions.append(region)
        else:
            merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], region[1]),
                                  merged_regions[-1][2] + region[2])  # Sum the data values
        
    return merged_regions, total_sum

def find_local_maxima(data):
    return np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1

def select_top_maxima(times, values, n_maxima, time_range):
    sorted_indices = np.argsort(values)[::-1]
    selected_indices = []
    for idx in sorted_indices:
        if len(selected_indices) == n_maxima:
            break
        if not selected_indices or all(abs(times[idx] - times[i]) > time_range for i in selected_indices):
            selected_indices.append(idx)
    return np.sort(selected_indices)

def export_csv_1(cft, times, vels, filename):
    df = pd.DataFrame({'CFT': cft, 'Time': times, 'Velocity': vels})
    output_dir = 'moonquakes_data_sta_lta_approach'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file = os.path.join(output_dir, f'{filename}_data.csv')
    df.to_csv(file, index=False)
    
def export_csv_2(rectangles, filename):
    start = []
    end = []
    percentages = []
    for rect in rectangles[0]:  # Note: accessing the first element of rectangles
        start.append(rect[0])
        end.append(rect[1])
        percentages.append(rect[2])
    df = pd.DataFrame({'Start Time': start, 'End time': end, 'Percentages': percentages})
    output_dir = 'moonquakes_data_sta_lta_approach'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file = os.path.join(output_dir, f'{filename}_indexes.csv')
    df.to_csv(file, index=False)

def main():
    # Load data
    row = 39
    cat = pd.read_csv('space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv')  # Assuming the catalog is in a CSV file
    filename = cat['filename'].iloc[row]
    file = f'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/{filename}.mseed'
    
    # file = 'space_apps_2024_seismic_detection/data/lunar/test/data/S16_GradeA/xa.s16.00.mhz.1974-11-11HR00_evid00160.mseed'
    
    stream = read(file)

    # Filter trace
    minfreq, maxfreq = 0.5, 1.0
    stream_filt = stream.copy()
    stream_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
    trace_filt = stream_filt.traces[0].copy()
    trace_times_filt = trace_filt.times()
    trace_data_filt = trace_filt.data

    # Perform STA/LTA
    df = trace_filt.stats.sampling_rate
    sta_len, lta_len = 100, 1500
    cft = classic_sta_lta(trace_data_filt, int(sta_len * df), int(lta_len * df))

    # Find local maxima
    local_maxima = find_local_maxima(cft)

    # Select top N maxima
    n_maxima = 10  # Number of maxima to select
    time_range = 200  # Time range in seconds
    selected_maxima = select_top_maxima(trace_times_filt[local_maxima], cft[local_maxima], n_maxima, time_range)

    # Plot results
    merged_rectangles = get_data_regions(trace_times_filt, local_maxima[selected_maxima], cft)

    export_csv_1(cft, trace_times_filt, trace_data_filt, filename)
    export_csv_2(merged_rectangles, filename)

if __name__ == "__main__":
    main()