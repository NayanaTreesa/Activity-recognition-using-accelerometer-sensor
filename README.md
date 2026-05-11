# Activity Recognition using Accelerometer Data

Machine learning pipeline for classifying human activities from smartphone accelerometer signals using time-domain feature extraction and a Multi-Layer Perceptron (MLP) Neural Network.

## Overview

This project processes raw 3-axis accelerometer data to distinguish between three activity classes:

| Class | Activity |
|-------|----------|
| 1 | Standing |
| 2 | Walking |
| 3 | Sitting |

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~98% |
| Test Accuracy | ~96% |
| Features per sample | 15 |
| Classifier | MLP Neural Network |

## Pipeline

```
Raw CSV files (x, y, z)
        │
        ▼
Feature Extraction (15 time-domain features)
        │
        ▼
Feature Visualisation
        │
        ▼
MLP Classifier Training
        │
        ▼
Evaluation (Confusion Matrix + Classification Report)
```

## Features Extracted

For each accelerometer window, 15 features are computed:

- **Statistical**: Mean, Min, Max, Standard Deviation for X, Y, Z axes (12 features)
- **Signal Magnitude Area (SMA)**: Overall movement intensity
- **Signal Vector Magnitude (SVM)**: Combined acceleration magnitude
- **Tilt Angle (TA)**: Posture estimation from Y-axis

## Project Structure

```
activity-recognition/
│
├── myFunctions.py          # Feature extraction helper functions
├── feature_extraction.py   # Step 1: Load data, extract features, save JSON
├── train_classifier.py     # Step 2: Train MLP, evaluate, save model
│
├── sevDB1/                 # Raw accelerometer dataset
│   ├── standing/
│   ├── walking/
│   └── sitting/
│
├── FeatureData.json        # Extracted feature matrix
├── Targets.json            # Class labels
├── myMLPClassifier.pkl     # Saved trained model
└── scaler.pkl              # Saved feature scaler
```

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/activity-recognition-accelerometer.git
cd activity-recognition-accelerometer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Extract features from raw data**
```bash
python feature_extraction.py
```

**4. Train and evaluate the classifier**
```bash
python train_classifier.py
```

## Requirements

```
numpy
scikit-learn
matplotlib
joblib
```

Or install via:
```bash
pip install -r requirements.txt
```

## Dataset

Raw accelerometer data collected from a smartphone, organised into per-activity folders. Each CSV file contains timestamped X, Y, Z acceleration readings for one activity window.

## Technologies

- **Language**: Python 3
- **ML Library**: scikit-learn (MLPClassifier)
- **Data Processing**: NumPy
- **Visualisation**: Matplotlib


Nayana Treesa Michael  
M.Sc. Information and Communication Engineering, TU Darmstadt  
[LinkedIn](https://linkedin.com/in/nayana-treesa-michael-4bb49a169)
