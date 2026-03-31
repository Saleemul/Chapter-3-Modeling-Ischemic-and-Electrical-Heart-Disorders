"""
Supplementary Code 2: ECG Classification with Gradient Boosting
Chapter 10 - Modeling Ischemic and Electrical Heart Disorders

Demonstrates:
  - Generating a labeled ECG feature dataset (3 classes)
  - Training a Gradient Boosting classifier with patient-level splits
  - Evaluation: ROC curves, confusion matrix, calibration curve
  - Feature importance via permutation importance

Requirements: numpy, pandas, scikit-learn, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ----------------------------------------------------------------
# 1. Generate synthetic labeled dataset (simulating extracted ECG features)
# ----------------------------------------------------------------
def generate_dataset(n_patients=500):
    """
    Create a synthetic dataset of ECG-derived features for 3 conditions.
    Each patient contributes 1-3 recordings (to test patient-level splits).
    """
    records = []
    patient_id = 0
    labels_map = {0: 'Normal', 1: 'Ischemia', 2: 'Arrhythmia'}

    for label in [0, 1, 2]:
        n_pat = n_patients // 3
        for _ in range(n_pat):
            n_recs = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            for _ in range(n_recs):
                # Overlapping distributions to produce realistic (imperfect) classification
                if label == 0:  # Normal
                    feats = {
                        'mean_hr': 74 + np.random.randn() * 16,
                        'hr_variability': 18 + np.random.randn() * 10,
                        'st_deviation': 0.0 + np.random.randn() * 0.15,
                        'qrs_duration': 92 + np.random.randn() * 16,
                        'qt_interval': 400 + np.random.randn() * 35,
                        'p_wave_present': np.random.choice([0, 1], p=[0.08, 0.92]),
                        'rr_irregularity': 0.05 + abs(np.random.randn()) * 0.05,
                        'age': 55 + np.random.randn() * 15,
                        'troponin': 0.03 + abs(np.random.randn()) * 0.06,
                    }
                elif label == 1:  # Ischemia
                    feats = {
                        'mean_hr': 80 + np.random.randn() * 17,
                        'hr_variability': 14 + np.random.randn() * 9,
                        'st_deviation': -0.10 + np.random.randn() * 0.16,
                        'qrs_duration': 96 + np.random.randn() * 16,
                        'qt_interval': 412 + np.random.randn() * 36,
                        'p_wave_present': np.random.choice([0, 1], p=[0.10, 0.90]),
                        'rr_irregularity': 0.06 + abs(np.random.randn()) * 0.05,
                        'age': 63 + np.random.randn() * 13,
                        'troponin': 0.08 + abs(np.random.randn()) * 0.10,
                    }
                else:  # Arrhythmia
                    feats = {
                        'mean_hr': 85 + np.random.randn() * 22,
                        'hr_variability': 25 + np.random.randn() * 14,
                        'st_deviation': -0.02 + np.random.randn() * 0.15,
                        'qrs_duration': 105 + np.random.randn() * 22,
                        'qt_interval': 425 + np.random.randn() * 38,
                        'p_wave_present': np.random.choice([0, 1], p=[0.35, 0.65]),
                        'rr_irregularity': 0.09 + abs(np.random.randn()) * 0.08,
                        'age': 60 + np.random.randn() * 15,
                        'troponin': 0.04 + abs(np.random.randn()) * 0.07,
                    }
                feats['patient_id'] = patient_id
                feats['label'] = label
                records.append(feats)
            patient_id += 1

    return pd.DataFrame(records)

# ----------------------------------------------------------------
# 2. Train and evaluate with patient-level cross-validation
# ----------------------------------------------------------------
df = generate_dataset(n_patients=600)

feature_cols = ['mean_hr', 'hr_variability', 'st_deviation', 'qrs_duration',
                'qt_interval', 'p_wave_present', 'rr_irregularity', 'age',
                'troponin']
X = df[feature_cols].values
y = df['label'].values
groups = df['patient_id'].values

print(f"Dataset: {len(df)} recordings from {df['patient_id'].nunique()} patients")
print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}\n")

# Patient-level GroupKFold (no leakage)
gkf = GroupKFold(n_splits=5)
y_true_all, y_prob_all = [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Confirm no patient leakage
    train_pats = set(groups[train_idx])
    test_pats = set(groups[test_idx])
    assert len(train_pats & test_pats) == 0, "Patient leakage detected!"

    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)
    y_true_all.append(y_test)
    y_prob_all.append(probs)

    # Per-fold macro AUC
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fold_auc = roc_auc_score(y_test_bin, probs, multi_class='ovr',
                             average='macro')
    print(f"  Fold {fold+1}: macro AUROC = {fold_auc:.3f}  "
          f"(train={len(train_idx)}, test={len(test_idx)})")

y_true_all = np.concatenate(y_true_all)
y_prob_all = np.vstack(y_prob_all)
y_pred_all = np.argmax(y_prob_all, axis=1)

# ----------------------------------------------------------------
# 3. Overall metrics
# ----------------------------------------------------------------
print("\n=== Classification Report (pooled across folds) ===")
print(classification_report(y_true_all, y_pred_all,
                            target_names=['Normal', 'Ischemia', 'Arrhythmia']))

y_true_bin = label_binarize(y_true_all, classes=[0, 1, 2])
overall_auc = roc_auc_score(y_true_bin, y_prob_all, multi_class='ovr',
                            average='macro')
print(f"Overall macro AUROC: {overall_auc:.3f}")

# ----------------------------------------------------------------
# 4. Plots
# ----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 4a. ROC curves
class_names = ['Normal', 'Ischemia', 'Arrhythmia']
colors = ['#0D9488', '#EA580C', '#2E5C8A']
for i, (name, color) in enumerate(zip(class_names, colors)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_all[:, i])
    roc_auc_val = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC={roc_auc_val:.2f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('(a) ROC Curves')
axes[0].legend(fontsize=9)

# 4b. Confusion matrix
cm = confusion_matrix(y_true_all, y_pred_all)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(ax=axes[1], cmap='Blues', colorbar=False)
axes[1].set_title('(b) Confusion Matrix')

# 4c. Calibration curve (ischemia class)
prob_true, prob_pred = calibration_curve(y_true_bin[:, 1], y_prob_all[:, 1],
                                         n_bins=8, strategy='uniform')
axes[2].plot(prob_pred, prob_true, 's-', color='#EA580C', lw=2,
             label='Ischemia class')
axes[2].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
axes[2].set_xlabel('Mean Predicted Probability')
axes[2].set_ylabel('Fraction of Positives')
axes[2].set_title('(c) Calibration Curve')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('/home/claude/supplementary_code/fig_classification_results.png',
            dpi=150, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------
# 5. Feature importance (permutation-based)
# ----------------------------------------------------------------
# Refit on full data for importance
clf_full = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42
)
clf_full.fit(X, y)

result = permutation_importance(clf_full, X, y, n_repeats=10,
                                random_state=42, scoring='accuracy')
sorted_idx = result.importances_mean.argsort()[::-1]

fig, ax = plt.subplots(figsize=(8, 5))
feature_labels = np.array(feature_cols)
ax.barh(range(len(sorted_idx)), result.importances_mean[sorted_idx],
        xerr=result.importances_std[sorted_idx],
        color='#2E5C8A', alpha=0.85)
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(feature_labels[sorted_idx])
ax.invert_yaxis()
ax.set_xlabel('Permutation Importance (decrease in accuracy)')
ax.set_title('Feature Importance for ECG Classification')
plt.tight_layout()
plt.savefig('/home/claude/supplementary_code/fig_feature_importance.png',
            dpi=150, bbox_inches='tight')
plt.close()

print("\nTop features by importance:")
for i in sorted_idx[:5]:
    print(f"  {feature_cols[i]:20s}  {result.importances_mean[i]:.4f} "
          f"+/- {result.importances_std[i]:.4f}")

print("\nFigures saved: fig_classification_results.png, fig_feature_importance.png")
