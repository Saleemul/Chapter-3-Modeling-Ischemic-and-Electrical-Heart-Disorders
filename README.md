Supplementary Material: Modeling Ischemic and Electrical Heart Disorders
Author: Saleem Ullah

Affiliation: Center of Engineering, Modeling and Applied Social Sciences, Universidade Federal do ABC, São Bernardo do Campo, Brazil

This repository contains the reproducible Python codebase accompanying the book chapter Modeling Ischemic and Electrical Heart Disorders. The scripts bridge clinical theory and computational practice, demonstrating automated electrocardiogram (ECG) processing, machine learning classification pipelines, and physics-informed interpolation for electrophysiology.

Rather than relying on synthetic or dummy data, the classification pipelines in this repository directly interface with the PTB-XL database via PhysioNet, streaming real patient records to ensure high clinical fidelity and authentic model evaluation.

🛠 Prerequisites and Installation
To execute these scripts locally, ensure you have Python 3.9 or higher installed.

Install the required dependencies using pip:

Bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn wfdb shap
Note: The wfdb library is required to stream physiological data from PhysioNet, and an active internet connection is necessary when running Scripts 1 and 2.

📂 Repository Structure and Scripts
1. code1_ecg_preprocessing.py
Description: Demonstrates the foundational steps of clinical signal processing. This script dynamically fetches real patient recordings (Normal Sinus Rhythm and Myocardial Infarction) from the PTB-XL database. It applies a 4th-order Butterworth bandpass filter to remove baseline wander and high-frequency noise, followed by dynamic thresholding for R-peak detection.

Outputs Generated: fig_ecg_preprocessing.png

2. code2_classification_pipeline.py
Description: A complete, end-to-end machine learning pipeline. It samples 500 real clinical records, extracts morphological and heart rate variability features, and trains a Gradient Boosting Classifier. Crucially, it implements strict patient-level GroupKFold cross-validation to prevent data leakage. The script evaluates performance via AUROC and classification reports, and interprets the model's decision-making using SHapley Additive exPlanations (SHAP).

Outputs Generated: * fig_worked_example_cm.png (Illustrative Confusion Matrix)

fig_real_shap_summary.png (SHAP Feature Importance Plot)

3. code3_physics_informed.py
Description: Illustrates the advantage of embedding physiological constraints into computational models. It reconstructs a cardiac conduction velocity (CV) field from sparse, noisy electrode measurements. By applying a smoothness constraint (a Laplacian penalty motivated by the diffusion equation), the model significantly reduces oscillatory artifacts compared to unconstrained cubic spline interpolation, accurately recovering the CV dip in a simulated myocardial scar region.

Outputs Generated: fig_physics_informed.png

🚀 Usage
You can run the scripts sequentially from your terminal. Because Scripts 1 and 2 stream data directly from PhysioNet, the initial execution may take a minute depending on your connection speed.

Bash
python code1_ecg_preprocessing.py
python code2_classification_pipeline.py
python code3_physics_informed.py
For immediate, browser-based execution without local environment setup, use the interactive Google Colab Notebook.

📊 Public Datasets Referenced
The models and methodologies discussed in the chapter and utilized in this repository draw upon the following open-access clinical databases:

PTB-XL (Used in codebase): A large publicly available electrocardiography dataset.

Link: https://physionet.org/content/ptb-xl/1.0.3/

Citation: Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7(154).

INCART: St. Petersburg INCART 12-lead Arrhythmia Database.

Link: https://physionet.org/content/incartdb/1.0.0/

MIMIC-IV: Medical Information Mart for Intensive Care IV.

Link: https://physionet.org/content/mimiciv/2.2/

Chapman-Shaoxing: A 12-lead electrocardiogram database for arrhythmia research.

Link: https://figshare.com/collections/ChapmanECG/4560497/2
