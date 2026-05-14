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

Repository Structure and Scripts
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

Usage
You can run the scripts sequentially from your terminal. Because Scripts 1 and 2 stream data directly from PhysioNet, the initial execution may take a minute depending on your connection speed.

Bash
python code1_ecg_preprocessing.py
python code2_classification_pipeline.py
python code3_physics_informed.py
For immediate, browser-based execution without local environment setup, use the interactive Google Colab Notebook.
## References and Datasets

The models, methodologies, and scripts in this repository are built on foundational research and open-access clinical databases. If you adapt this code for your own work, please ensure you cite the appropriate primary sources below.

### Public Clinical Datasets
The following databases provide the clinical signals and metadata discussed in the chapter and utilized in the code:

* **PTB-XL** (Used in the classification pipeline): A large publicly available 12-lead electrocardiography dataset. [View Dataset](https://physionet.org/content/ptb-xl/1.0.3/)
* **INCART**: St. Petersburg INCART 12-lead Arrhythmia Database. [View Dataset](https://physionet.org/content/incartdb/1.0.0/)
* **MIMIC-IV**: Medical Information Mart for Intensive Care IV. [View Dataset](https://physionet.org/content/mimiciv/2.2/)
* **Chapman-Shaoxing**: A 12-lead electrocardiogram database for arrhythmia research. [View Dataset](https://figshare.com/collections/ChapmanECG/4560497/2)

### Core Citations

**Primary Datasets and Repositories**
* Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7(154). https://doi.org/10.1038/s41597-020-0495-6
* Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215-e220. https://doi.org/10.1161/01.CIR.101.23.e215

**Machine Learning and Interpretability**
* Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30, 4765-4774.
* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

**Physics-Informed Modeling**
* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. https://doi.org/10.1016/j.jcp.2018.10.045
* Sahli Costabal, F., Yang, Y., Perdikaris, P., Hurtado, D. E., & Kuhl, E. (2020). Physics-informed neural networks for cardiac activation mapping. *Frontiers in Physics*, 8, 42. https://doi.org/10.3389/fphy.2020.00042

**Clinical Frameworks (CiPA and CRT)**
* Colatsky, T., Fermini, B., Gintant, G., Pierson, J. B., Sager, P., Sekino, Y., ... & Stockbridge, N. (2016). The Comprehensive in Vitro Proarrhythmia Assay (CiPA) initiative: Update on progress. *Journal of Pharmacological and Toxicological Methods*, 81, 15-20.
* Cikes, M., Sanchez-Martinez, S., Claggett, B., Duchateau, N., Piella, G., Butakoff, C., ... & Bijnens, B. (2019). Machine learning-based phenogrouping in heart failure to identify responders to cardiac resynchronization therapy. *European Journal of Heart Failure*, 21(1), 74-85.


Link: https://figshare.com/collections/ChapmanECG/4560497/2
