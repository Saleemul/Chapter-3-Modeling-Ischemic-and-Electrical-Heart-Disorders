"""
Supplementary Code 1: ECG Preprocessing and Feature Extraction
Chapter: Modeling Ischemic and Electrical Heart Disorders

Demonstrates:
  - Streaming real ECG data from the PTB-XL database via PhysioNet
  - Bandpass filtering and baseline correction
  - QRS detection via dynamic peak finding
"""

import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings('ignore')

def bandpass_filter(data, fs=100, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

def fetch_and_process_record(record_name, dir_path, title, ax):
    print(f"Fetching {title}...")
    remote_dir = f"ptb-xl/1.0.3/{dir_path}"
    record = wfdb.rdrecord(record_name, pn_dir=remote_dir)
    
    # Extract Lead I
    raw_signal = record.p_signal[:, 0]
    t = np.arange(len(raw_signal)) / record.fs
    
    clean_signal = bandpass_filter(raw_signal, fs=record.fs)
    threshold = np.max(clean_signal) * 0.4
    peaks, _ = find_peaks(clean_signal, height=threshold, distance=int(0.4 * record.fs))
    
    ax.plot(t, raw_signal, alpha=0.3, label='Raw (Lead I)', color='gray')
    ax.plot(t, clean_signal, label='Filtered', color='#2E5C8A')
    ax.plot(t[peaks], clean_signal[peaks], 'rv', markersize=6, label='R-peaks')
    
    ax.set_ylabel('mV')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 5) # Show first 5 seconds

if __name__ == '__main__':
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    # Record 00001_lr is a Normal Sinus Rhythm
    fetch_and_process_record('00001_lr', 'records100/00000', 'Normal Sinus Rhythm', axes[0])
    
    # Record 00008_lr has an MI label
    fetch_and_process_record('00008_lr', 'records100/00000', 'Myocardial Infarction', axes[1])
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('fig_ecg_preprocessing.png', dpi=300)
    print("Saved fig_ecg_preprocessing.png")
