# Supplementary Code
## Chapter 10: Modeling Ischemic and Electrical Heart Disorders

Three self-contained Python scripts accompany this chapter. Each generates
its own figures and prints results to the console.

### Requirements

```
Python >= 3.9
numpy >= 1.24
scipy >= 1.10
pandas >= 1.5
matplotlib >= 3.6
scikit-learn >= 1.2
```

Install: `pip install numpy scipy pandas matplotlib scikit-learn`

### Scripts

| Script | Description | Output |
|--------|-------------|--------|
| `code1_ecg_preprocessing.py` | Synthetic ECG generation (normal, ischemia, AF), bandpass filtering, R-peak detection, feature extraction | `fig_ecg_preprocessing.png` |
| `code2_classification_pipeline.py` | Gradient boosting classification with patient-level GroupKFold, ROC curves, calibration, permutation importance | `fig_classification_results.png`, `fig_feature_importance.png` |
| `code3_physics_informed.py` | Physics-constrained vs data-only interpolation for conduction velocity mapping from sparse electrodes | `fig_physics_informed.png` |

### Running

```bash
python code1_ecg_preprocessing.py
python code2_classification_pipeline.py
python code3_physics_informed.py
```

All scripts use `np.random.seed(42)` for reproducibility.

### Public Datasets (not bundled)

- **PTB-XL**: https://physionet.org/content/ptb-xl/1.0.3/
- **INCART**: https://physionet.org/content/incartdb/1.0.0/
- **MIMIC-IV**: https://physionet.org/content/mimiciv/2.2/
- **Chapman-Shaoxing**: https://figshare.com/collections/ChapmanECG/4560497/2

### License

Released under the MIT License for academic use.
