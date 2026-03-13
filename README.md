# Escalator Predictive Maintenance & Anomaly Detection

This project focuses on simulating realistic operational data for subway escalators and building machine learning models to predict potential hardware failures or anomalies before they lead to breakdowns. By combining physical simulations with both supervised and unsupervised learning techniques, the project creates a robust anomaly detection system.

## 🚀 Key Features

*   **Realistic Data Simulation:** Simulates physical wear-and-tear (aging), passenger loads across different stations (잠실, 고속터미널, 용두), seasonal temperature variations, electrical current, and vibration data.
*   **Hard Mode Scenarios:** Generates rare and complex failure cases to rigorously test the models' discriminative power.
*   **Supervised Learning:** Utilizes XGBoost to classify potential failures based on engineered features.
*   **Unsupervised Learning:** Implements Isolation Forest to detect unknown operational anomalies without predefined labels.
*   **Hybrid Ensemble Engine:** Combines the strengths of supervised predictions (70%) and unsupervised risk scores (30%) to flag escalators for urgent inspection.
*   **BI Integration:** Exports processed data for visual analytics in Tableau (`tableau_escalator_v6_final.csv`).

## 📁 Repository Structure & Workflow

The core workflow consists of data generation, model training, and hybrid evaluation.

### 1. Data Generation & Preprocessing (`v6Data.py`)
This script acts as the core simulator. It:
*   Loads base passenger data (`시간대별 승차인원.csv`).
*   Iterates through different stations and escalator lengths over a multi-week period.
*   Calculates dynamic loads, aging impacts, motor current, vibration, and temperature characteristics based on passenger volume and physical principles.
*   Generates probabilistic fault labels (`Label`=1).
*   Applies **Feature Engineering** (e.g., `Rel_Current`, `Load_Per_Pax`, diffs) and **Robust Scaling**.
*   Outputs: `escalator_v6_pro.csv`, `final_preprocessed_v6.csv`, and saves the scaler (`robust_scaler_v5.pkl`).

### 2. Supervised Learning: XGBoost (`xgBoost.py`)
Utilizes the preprocessed `v6` data to predict failures.
*   Filters for operational hours (5 AM to 1 AM).
*   Handles extreme data imbalance adjusting `scale_pos_weight`.
*   Optimized for the Area Under the Precision-Recall Curve (`aucpr`) to reduce false positives.
*   Outputs: `xgboost_model_v6_final.pkl`.

### 3. Unsupervised Learning: Isolation Forest (`isolationForest.py`)
Provides a safety net by identifying novel anomalies that the supervised model might miss.
*   Learns solely from the distribution of normal operational features.
*   Calculates an anomaly score to detect deviations.
*   Outputs: `iso_forest_v6.pkl`.

### 4. Hybrid Risk Assessment (`hybridModel.py`)
Loads the trained XGBoost and Isolation Forest models.
*   Calculates a normalized probability score from XGBoost (0.0 - 1.0).
*   Calculates a normalized risk score from Isolation Forest (0.0 - 1.0).
*   Calculates `Final_Risk = (XGB_Prob * 0.7) + (IF_Risk * 0.3)`.
*   Flags operations with a `Final_Risk` > 0.8 as requiring **urgent inspection**.

### Jupyter Notebooks
*   `escalator.ipynb`: General EDA and simulator prototyping.
*   `machineLearning.ipynb`: Exploratory ML modeling and evaluation.
*   `threshold_model.ipynb`: Tests baseline rule-based threshold approaches.

## ⚙️ Setup & Execution

1.  **Generate Data**: Run `v6Data.py` to simulate the data and create the preprocessed outputs. *(Ensure you have the required base passenger CSV in the correct path).*
2.  **Train Models**: 
    *   Run `xgBoost.py` to train the supervised classifier.
    *   Run `isolationForest.py` to train the unsupervised anomaly detector.
3.  **Evaluate Risk**: Run `hybridModel.py` to combine the models and view instances requiring urgent attention.

