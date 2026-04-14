import numpy as np
import scipy.signal as signal
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
from datetime import datetime
import os
from matplotlib.patches import Patch
import json
from scipy.signal import resample_poly

def resample_signal_poly(signal, original_rate=100, target_rate=125):
    from math import gcd
    factor = gcd(original_rate, target_rate)
    up = target_rate // factor
    down = original_rate // factor
    return resample_poly(signal, up, down)

def pan_tompkins(ecg, fs=125):
    """
    Pan-Tompkins R-peak detection algorithm.
    ecg: 1D numpy array of raw ECG signal
    fs: Sampling frequency in Hz
    """
    
    # 1. Bandpass Filter (5-15 Hz)
    nyq = 0.5 * fs
    low = 5 / nyq
    high = 15 / nyq
    b, a = signal.butter(1, [low, high], btype='band')
    ecg_filtered = signal.filtfilt(b, a, ecg)
    
    # 2. Differentiation
    diff = np.ediff1d(ecg_filtered)
    
    # 3. Squaring
    squared = diff ** 2
    
    # 4. Moving Window Integration (≈150ms window)
    window_size = int(0.150 * fs)
    mwa = np.convolve(squared, np.ones(window_size) / window_size, mode='same')
    
    # 5. Adaptive Thresholding
    threshold = np.mean(mwa) * 1.2  # tweak this as needed
    peaks, _ = signal.find_peaks(mwa, height=threshold, distance=int(0.3 * fs))  # refractory period ≈300ms

    # 6. Refine R-peaks: Find max in original ECG near each mwa peak
    r_peaks = []
    search_window = int(0.1 * fs)  # ±100ms window
    for peak in peaks:
        start = max(peak - search_window, 0)
        end = min(peak + search_window, len(ecg))
        r_peak = start + np.argmax(ecg[start:end])
        r_peaks.append(r_peak)

    return np.array(r_peaks), mwa, ecg_filtered


def extract_beats_from_r(ecg, r_peaks, fs=125, post_ms=400):
    post_samples = int(fs)
    beats = []
    for r in r_peaks:
        end = r + post_samples
        if end <= len(ecg):
            beat = ecg[r:end]
            beats.append(beat)
    return np.array(beats)

def pad_beats_to_187(beats, target_length=187):
    """
    Pads each beat with zeros at the end to reach target length.
    Input: beats = list or array of shape (num_beats, current_length)
    Output: padded_beats = array of shape (num_beats, target_length)
    """
    padded_beats = []

    for beat in beats:
        pad_width = target_length - len(beat)
        if pad_width > 0:
            beat_padded = np.pad(beat, (0, pad_width), mode='constant', constant_values=0)
        else:
            beat_padded = beat[:target_length]  # truncate if longer
        padded_beats.append(beat_padded)

    return np.array(padded_beats)

def normalize_min_max(beats):
    """
    Normalize ECG beats to the range [0, 1] using Min-Max normalization.
    Input: beats = numpy array of shape (num_beats, 187)
    Output: normalized_beats = Min-Max normalized ECG beats
    """
    # Min and Max for each beat
    min_vals = np.min(beats, axis=1, keepdims=True)
    max_vals = np.max(beats, axis=1, keepdims=True)
    
    # Normalize each beat (avoid division by zero if max equals min)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # prevent division by zero
    
    normalized_beats = (beats - min_vals) / range_vals

    return normalized_beats

def save_ecg_signal(ecg, r_peaks, save_name = "", save_path="./ecg_figure/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    uuid = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # Plot result
    plt.figure(figsize=(10, 6)) 
    plt.plot(ecg, label=f'ECG signal', alpha=0.5)
    plt.plot(r_peaks, ecg[r_peaks], 'ro', label='R-peaks')
    plt.title(f'ECG Record at {uuid}')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    # Save the plot to a file (e.g., PNG)
    plt.savefig(f'{save_path}/ecg_plot{save_name}_{uuid}.png', dpi=300)  # You can adjust the DPI for better resolution

def save_ecg_prediction(ecg, r_peaks, labels, save_name = "", save_path="./ecg_figure"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    uuid = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    label_map = {0: "N", 1: "S", 2: "P", 3: "F", 4: "U"}
    # Color map for labels 1-4
    label_colors = {
        1: 'red',
        2: 'green',
        3: 'orange',
        4: 'purple'
    }
    # Create a figure and axis
    plt.figure(figsize=(10, 6))
    # Plot the ECG signal
    plt.plot(ecg, label="ECG Signal", color='blue')
    # Width of each bounding box (in number of samples)
    box_width = 40
    signal_length = len(ecg)
    for i, r in enumerate(r_peaks):
        label = labels[i] if i < len(labels) else 0
        if label == 0:
            continue
        color = label_colors.get(label, 'black')
        if(signal_length > r+box_width):
            # Draw a shaded box starting at R-peak and extending box_width samples
            plt.axvspan(r, r + box_width, color=color, alpha=0.3, label=f'Label {label}' if f'Label {label}' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Create custom legend patches
    legend_patches = [
        Patch(color=color, label=label_map[label])
        for label, color in label_colors.items()
    ]
    plt.title(f"ECG Signal Predict at {uuid}")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    # Legend below the plot
    plt.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
           fancybox=True, shadow=True, ncol=len(legend_patches))
    plt.tight_layout()
    plt.grid(True)
    ecg_figure = f'ecg_prediction{save_name}_{uuid}.png'
    plt.savefig(f'{save_path}/{ecg_figure}', dpi=300)  # You can adjust the DPI for better resolution
    
    data_to_save = {
    "ecg_signal": ecg.tolist(),
    "r_peaks": r_peaks.tolist(),
    "labels": labels.tolist()
    }
    # Save to a JSON file
    with open(f"{save_path}/ecg_prediction_{uuid}.json", "w") as f:
        json.dump(data_to_save, f, indent=2)
    return ecg_figure

def save_prediction_per_beat(input,output, save_name = "", save_path="./ecg_figure/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    uuid = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    label_map = {0: 'N', 1: 'S', 2: 'P', 3: 'F', 4: 'U'}
    input_np = input
    output_np = output
    plt.figure(figsize=(24,12))
    for i in range(len(input_np)):
        plt.subplot(4,4,i+1)
        plt.plot(input_np[i])
        plt.title(f"Prediction {label_map[output_np[i]]}")
        plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig(f'{save_path}/ecg_ber_beat{save_name}_{uuid}.png', dpi=300)  # You can adjust the DPI for better resolution
    
def main_ecg_processing(ecg, sampling_rate=125):
    r_peaks, mwa, ecg_filtered = pan_tompkins(ecg, sampling_rate)
    save_ecg_signal(ecg,r_peaks)
    beats = extract_beats_from_r(ecg, r_peaks, sampling_rate)
    normalize_beats = normalize_min_max(beats)
    padded_beats = pad_beats_to_187(normalize_beats)
    return padded_beats, r_peaks