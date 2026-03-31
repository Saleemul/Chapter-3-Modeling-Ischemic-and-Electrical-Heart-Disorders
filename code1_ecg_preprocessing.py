"""
Supplementary Code 1: ECG Preprocessing and Feature Extraction
Chapter 10 - Modeling Ischemic and Electrical Heart Disorders

Demonstrates:
  - Synthetic ECG generation (normal, ischemic, arrhythmia)
  - Bandpass filtering and baseline correction
  - QRS detection via simple peak finding
  - Feature extraction (ST deviation, QRS width, HR, HRV)

Requirements: numpy, matplotlib, scipy, pandas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import os

np.random.seed(42)

# --- 1. Synthetic ECG generation ---
def synthetic_ecg_beat(fs=500, hr=72, ischemia=False, af=False):
    """Generate one cycle of synthetic ECG."""
    T = 60.0 / hr
    t = np.arange(0, T, 1/fs)
    ecg = np.zeros_like(t)

    # P wave
    if not af:
        p_center = 0.15 * T
        ecg += 0.15 * np.exp(-((t - p_center) / 0.02) ** 2)

    # QRS complex
    qrs_center = 0.4 * T
    ecg += -0.15 * np.exp(-((t - (qrs_center - 0.015)) / 0.006) ** 2)  # Q
    ecg += 1.2 * np.exp(-((t - qrs_center) / 0.008) ** 2)              # R
    ecg += -0.3 * np.exp(-((t - (qrs_center + 0.015)) / 0.006) ** 2)   # S

    # ST segment + T wave
    t_center = 0.65 * T
    if ischemia:
        # ST depression for ischemia
        st_start, st_end = qrs_center + 0.03, t_center - 0.04
        mask = (t >= st_start) & (t <= st_end)
        ecg[mask] -= 0.2
        ecg += 0.25 * np.exp(-((t - t_center) / 0.04) ** 2)  # flattened T
    else:
        ecg += 0.35 * np.exp(-((t - t_center) / 0.04) ** 2)  # normal T

    return t, ecg


def generate_ecg_strip(duration=10, fs=500, hr=72, ischemia=False, af=False):
    """Chain beats into a continuous strip, add noise."""
    total_samples = int(duration * fs)
    signal = np.zeros(total_samples)
    idx = 0

    while idx < total_samples:
        if af:
            beat_hr = hr + np.random.randint(-30, 30)  # irregular RR
            beat_hr = max(50, min(150, beat_hr))
        else:
            beat_hr = hr + np.random.randn() * 2
        t_beat, beat = synthetic_ecg_beat(fs=fs, hr=beat_hr,
                                          ischemia=ischemia, af=af)
        n = len(beat)
        if idx + n > total_samples:
            n = total_samples - idx
        signal[idx:idx+n] = beat[:n]
        idx += n

    # Add realistic noise
    signal += 0.02 * np.random.randn(total_samples)           # EMG noise
    signal += 0.03 * np.sin(2 * np.pi * 0.3 *
                            np.arange(total_samples) / fs)     # baseline wander
    return np.arange(total_samples) / fs, signal


# --- 2. Preprocessing ---
def bandpass_filter(signal, fs=500, low=0.5, high=40.0, order=4):
    """Apply a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)


# --- 3. QRS detection and feature extraction ---
def extract_features(signal, fs=500):
    """Extract cardiac features from a preprocessed ECG strip."""
    # R-peak detection
    peaks, props = find_peaks(signal, height=0.4, distance=int(0.4 * fs))

    if len(peaks) < 3:
        return None

    rr_intervals = np.diff(peaks) / fs  # in seconds
    hr_bpm = 60.0 / rr_intervals

    features = {
        'mean_hr_bpm': np.mean(hr_bpm),
        'std_hr_bpm': np.std(hr_bpm),
        'rmssd_ms': np.sqrt(np.mean(np.diff(rr_intervals * 1000) ** 2)),
        'mean_r_amplitude': np.mean(signal[peaks]),
        'num_beats': len(peaks),
    }

    # ST-segment deviation: sample 80ms after each R-peak
    st_offset = int(0.08 * fs)
    st_vals = []
    for pk in peaks:
        idx = pk + st_offset
        if idx < len(signal):
            st_vals.append(signal[idx])
    features['mean_st_level'] = np.mean(st_vals) if st_vals else 0.0
    features['std_st_level'] = np.std(st_vals) if st_vals else 0.0

    return features


# --- 4. Run demo ---
if __name__ == '__main__':
    fs = 500
    conditions = [
        ('Normal Sinus Rhythm', dict(hr=72, ischemia=False, af=False)),
        ('Ischemia (ST depression)', dict(hr=80, ischemia=True, af=False)),
        ('Atrial Fibrillation', dict(hr=90, ischemia=False, af=True)),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    all_features = []

    for i, (label, params) in enumerate(conditions):
        t, raw = generate_ecg_strip(duration=8, fs=fs, **params)
        filtered = bandpass_filter(raw, fs=fs)
        feats = extract_features(filtered, fs=fs)
        feats['condition'] = label
        all_features.append(feats)

        axes[i].plot(t[:2500], raw[:2500], alpha=0.3, label='Raw', color='gray')
        axes[i].plot(t[:2500], filtered[:2500], label='Filtered', color='#2E5C8A')

        # Mark R-peaks on filtered signal
        peaks, _ = find_peaks(filtered[:2500], height=0.4,
                              distance=int(0.4 * fs))
        axes[i].plot(t[peaks], filtered[peaks], 'rv', markersize=6,
                     label='R-peaks')
        axes[i].set_ylabel('mV')
        axes[i].set_title(label, fontsize=11, fontweight='bold')
        axes[i].legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig('/home/claude/supplementary_code/fig_ecg_preprocessing.png',
                dpi=150, bbox_inches='tight')
    plt.close()

    df = pd.DataFrame(all_features)
    print("\n=== Extracted Features ===")
    print(df.to_string(index=False))
    print("\nFigure saved: fig_ecg_preprocessing.png")
