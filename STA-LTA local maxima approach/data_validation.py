# Enter a catalog file and this code prints you how many arrival times are within the rectangles

import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta

def cft_validation(times, cft, selected_maxima, arrival_time, window_before=300, window_after=600):
    cft_values = cft[selected_maxima]
    cft_max = np.max(np.abs(cft_values))
    
    regions = []
    for idx, value in zip(selected_maxima, cft_values):
        start = max(0, times[idx] - window_before)
        end = times[idx] + window_after
        regions.append((start, end, value / cft_max))
    
    # Merge overlapping regions
    merged_regions = []
    for region in sorted(regions):
        if not merged_regions or merged_regions[-1][1] < region[0]:
            merged_regions.append(region)
        else:
            merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], region[1]),
                                  max(merged_regions[-1][2], region[2]))  # Sum the data values
    
    count = 0
    perc = 0
    for merg_reg in merged_regions:
        if merg_reg[0] <= arrival_time and arrival_time <= merg_reg[1]:
            count += 1
            perc += 1 * merg_reg[2]
        else:
            count += 0
            perc += 0
            
    return count, perc
    

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

def main():
    counter = 0
    percentage = 0
    
    # Load data
    cat = pd.read_csv(csv_catalog)  # Assuming the catalog is in a CSV file
    for row in range(len(cat)):
        filename = cat['filename'].iloc[row]
        arrival_time = cat['time_rel(sec)'].iloc[row]
        file = f'space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/{filename}.mseed'
        
        stream = read(file)

        # Filter trace
        minfreq, maxfreq = 0.1, 1
        stream_filt = stream.copy()
        stream_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
        trace_filt = stream_filt.traces[0].copy()
        trace_times_filt = trace_filt.times()
        trace_data_filt = trace_filt.data

        # Perform STA/LTA
        df = trace_filt.stats.sampling_rate
        sta_len, lta_len = 100, 1500
        cft = classic_sta_lta(trace_data_filt**2, int(sta_len * df), int(lta_len * df))
    
        # Find local maxima
        local_maxima = find_local_maxima(cft)

        # Select top N maxima
        n_maxima = 4000  # Number of maxima to select
        time_range = 1  # Time range in seconds
        selected_maxima = select_top_maxima(trace_times_filt[local_maxima], cft[local_maxima], n_maxima, time_range)

        # Plot results
        count, percent = cft_validation(trace_times_filt, cft, local_maxima[selected_maxima], arrival_time)
        counter += count
        percentage += percent
        
    print(counter)
    print(percentage)  # Counter[row] * Percent[row]

if __name__ == "__main__":
    main()
