"""
Supplementary Code 2: ECG Classification Pipeline
Chapter: Modeling Ischemic and Electrical Heart Disorders

Demonstrates:
  - Extracting features from 500 real PTB-XL patient records
  - Gradient Boosting classification with strict patient-level GroupKFold splits
  - Generating performance metrics, confusion matrix, and SHAP explanations
"""

import pandas as pd
import numpy as np
import wfdb
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def bandpass_filter(data, fs=100, low=0.5, high=40.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data)

if __name__ == '__main__':
    print("Loading PTB-XL metadata (this requires internet access)...")
    df = pd.read_csv('https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv('https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)

    def get_label(classes):
        if 'MI' in classes and 'NORM' not in classes: return 1
        if 'NORM' in classes and 'MI' not in classes: return 0
        return -1

    df['label'] = df.diagnostic_superclass.apply(get_label)
    df = df[df.label != -1]

    # Process 500 records for demonstration
    df = df.sample(n=500, random_state=42)

    features_list = []
    print(f"Streaming and extracting features from {len(df)} records. This takes a minute...")

    for idx, row in df.iterrows():
        try:
            file_path = row['filename_lr']
            parts = file_path.split('/')
            dir_path = '/'.join(parts[:-1])
            base_name = parts[-1]
            
            remote_dir = f"ptb-xl/1.0.3/{dir_path}"
            record = wfdb.rdrecord(base_name, pn_dir=remote_dir)
            raw_signal = record.p_signal[:, 0] 
            
            clean_signal = bandpass_filter(raw_signal)
            threshold = np.max(clean_signal) * 0.4
            peaks, _ = find_peaks(clean_signal, height=threshold, distance=40) 
            
            if len(peaks) < 3: continue

            rr_intervals = np.diff(peaks) / 100.0
            hr_bpm = 60.0 / rr_intervals
            st_offset = int(0.08 * 100)
            st_vals = [clean_signal[pk + st_offset] for pk in peaks if pk + st_offset < len(clean_signal)]

            features_list.append({
                'age': row['age'],
                'mean_hr': np.mean(hr_bpm),
                'hr_variability': np.std(hr_bpm),
                'rmssd': np.sqrt(np.mean(np.diff(rr_intervals * 1000) ** 2)),
                'st_deviation': np.mean(st_vals) if st_vals else 0.0,
                'patient_id': row['patient_id'],
                'label': row['label']
            })
        except Exception:
            continue

    features_df = pd.DataFrame(features_list)
    print(f"Extraction complete. Training model on {len(features_df)} valid records...")

    X = features_df.drop(columns=['patient_id', 'label']).values
    y = features_df['label'].values
    groups = features_df['patient_id'].values
    feature_names = features_df.drop(columns=['patient_id', 'label']).columns

    gkf = GroupKFold(n_splits=5)
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

    y_true_all, y_prob_all, y_pred_all = [], [], []

    for train_idx, test_idx in gkf.split(X, y, groups):
        clf.fit(X[train_idx], y[train_idx])
        y_true_all.extend(y[test_idx])
        y_prob_all.extend(clf.predict_proba(X[test_idx])[:, 1])
        y_pred_all.extend(clf.predict(X[test_idx]))

    print("\n=== Model Results ===")
    print(f"Overall AUROC: {roc_auc_score(y_true_all, y_prob_all):.3f}")
    print(classification_report(y_true_all, y_pred_all, target_names=['Normal', 'MI']))

    # Generate Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'MI'], yticklabels=['Normal', 'MI'])
    plt.ylabel('Actual Label'); plt.xlabel('Predicted Label')
    plt.title('Illustrative Confusion Matrix (PTB-XL Subset)')
    plt.tight_layout()
    plt.savefig('fig_worked_example_cm.png', dpi=300)

    # Generate SHAP Plot
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance (MI vs Normal)")
    plt.tight_layout()
    plt.savefig('fig_real_shap_summary.png', dpi=150, bbox_inches='tight')
    print("Saved outputs: fig_worked_example_cm.png, fig_real_shap_summary.png")
