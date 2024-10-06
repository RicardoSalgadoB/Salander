import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import classic_sta_lta
from matplotlib.patches import Rectangle

def plot_earthquake_regions(ax, times, selected_maxima, cft_values, cft_max, window_before=30, window_after=60):
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
                                  max(merged_regions[-1][2],region[2]))  # Sum the data values
    
    for start, end, value in merged_regions:
        rect = Rectangle((start, ax.get_ylim()[0]), end - start, ax.get_ylim()[1] - ax.get_ylim()[0],
                         facecolor='lightgray', edgecolor='red', alpha=0.5, zorder=1, linewidth=1)
        ax.add_patch(rect)
        relative_value = (value/cft_max)**3
        label = f'{relative_value*100:.2f}% chance this is a moon/marsquake'
        ax.text(end+20, ax.get_ylim()[0], label,
                horizontalalignment='center', verticalalignment='bottom', rotation=90, fontsize=6,
                color='red')
        
    return merged_regions

def plot_vels(trace_data_filt, trace_times_filt, selected_maxima, cft):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    ax.plot(trace_times_filt, trace_data_filt, label='Filtered Data', alpha = 0.25)
    
    cft_values = cft[selected_maxima]
    cft_max = np.max(np.abs(cft_values))
    
    # Plot the data first to set the y-limits
    y_min, y_max = ax.get_ylim()
    
    # Plot earthquake regions
    merged_regions = plot_earthquake_regions(ax, trace_times_filt, selected_maxima, cft_values, cft_max)
    
    # Export data for each merged rectangle
    # export_rectangle_data(trace_times_filt, trace_data_filt, merged_regions, filename)
    
    # Reset the y-limits to ensure the rectangles don't change the scale
    ax.set_xlim([min(trace_times_filt), max(trace_times_filt)])
    ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()

def plot_cft(trace_times_filt, cft, selected_maxima):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    
    # Plot CFT to set the initial y-axis limits
    ax.plot(trace_times_filt, cft, label='CFT', zorder=2, alpha = 0.25)
    
    cft_values = cft[selected_maxima]
    cft_max = np.max(np.abs(cft_values))
    
    # Get the y-axis limits
    y_min, y_max = ax.get_ylim()
    
    # Plot earthquake regions
    plot_earthquake_regions(ax, trace_times_filt, selected_maxima, cft_values, cft_max)
    
    # Reset the y-axis limits
    ax.set_ylim(y_min, y_max)
    
    # Plot selected maxima
    ax.plot(trace_times_filt[selected_maxima], cft[selected_maxima], 'ro', label='Selected Maxima', zorder=3)
    
    ax.set_xlim([0, max(trace_times_filt)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Characteristic function')
    ax.legend()
    plt.tight_layout()
    plt.show()

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
    # For training miniseed files
    row = i
    cat = pd.read_csv(catalog_file)  # Assuming the catalog is in a CSV file
    filename = cat['filename'].iloc[row]
    file = f'{input_directory}{filename}.mseed'

    # For all other miniseed files
    # file = input_file
    
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
    sta_len, lta_len = 60, 400
    cft = classic_sta_lta(trace_data_filt, int(sta_len * df), int(lta_len * df))
    
    # Find local maxima
    local_maxima = find_local_maxima(cft)

    # Select top N maxima
    n_maxima = 90  # Number of maxima to select
    time_range = 1  # Time range in seconds
    selected_maxima = select_top_maxima(trace_times_filt[local_maxima], cft[local_maxima], n_maxima, time_range)

    # Plot results
    plot_cft(trace_times_filt, cft, local_maxima[selected_maxima])
    plot_vels(trace_data_filt, trace_times_filt, local_maxima[selected_maxima], cft)

    return trace_times_filt[local_maxima[selected_maxima]], cft[local_maxima[selected_maxima]]

if __name__ == "__main__":
    main()
