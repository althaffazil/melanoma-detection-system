# Automated Melanoma Detection and Lesion Segmentation

An end-to-end deep learning system for automated skin lesion segmentation and melanoma classification, deployed using Streamlit Cloud.

This project integrates medical image segmentation (U-Net) and melanoma classification (EfficientNet) into a modular, production-ready application.


## 🚀 Live Application

🔗 **Streamlit Deployment:** https://melanoma-detection-system.streamlit.app


## 🩺 Project Overview

This system performs two core tasks:

### 1️⃣ Lesion Segmentation
- Model: U-Net
- Task: Pixel-wise lesion mask prediction
- Dice Score: ~0.92
- IoU: ~0.85

### 2️⃣ Melanoma Classification
- Backbone: EfficientNet
- Metric: ROC-AUC (~0.93–0.96)
- Class imbalance handled via `pos_weight`
- Decision threshold optimized using ROC analysis (Youden’s J statistic)

The deployed application allows users to:

- Upload dermoscopic images
- Visualize lesion segmentation overlay
- View melanoma probability score
- Receive model-based diagnostic decision


## 🏗️ System Architecture

### 🔹 Segmentation Module
- U-Net architecture
- BCE + Dice-based training
- Binary mask prediction
- Post-processing thresholding for clean overlays

### 🔹 Classification Module
- EfficientNet backbone
- Binary melanoma classification
- Stratified 80/20 train-validation split
- Mixed precision training (AMP)
- Cosine learning rate scheduler
- Threshold calibrated via ROC curve optimization

## 📂 Folder Structure

```

.
├── app/
│   └── streamlit_app.py        # Streamlit entry point (deployment)
│
├── checkpoints/
│   ├── classifier_best.pth     # Best trained classification model
│   └── segmentation_best.pth   # Best trained segmentation model
│
├── src/
│   ├── inference/
│   │   └── predictor.py        # Unified inference logic
│   │
│   ├── models/
│   │   ├── classifier.py       # EfficientNet classifier
│   │   └── unet.py             # U-Net segmentation model
│   │
│   ├── training/
│   │   ├── train_classifier.py
│   │   └── train_segmentation.py
│   │
│   ├── utils/
│   │   └── utils.py
│   │
│   └── config.py
│
├── requirements.txt
└── README.md

````


## ⚙️ Local Installation

Clone the repository:

```bash
git clone https://github.com/althaffazil/melanoma-detection-system.git
cd melanoma-detection-system
````

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app/streamlit_app.py
```



## 📊 Model Performance

### Classification (Melanoma Detection)

| Metric              | Value                  |
| ------------------- | ---------------------- |
| ROC-AUC             | ~0.94                  |
| Optimal Threshold   | ~0.25                  |
| Validation Strategy | Stratified 80/20 Split |

### Segmentation

| Metric     | Value |
| ---------- | ----- |
| Dice Score | ~0.92 |
| IoU        | ~0.85 |


## 🧠 Key Engineering Highlights

* Modular project architecture (training / inference separation)
* Reproducible stratified data splitting
* Class imbalance correction using weighted BCE loss
* Mixed precision training for GPU efficiency
* Cosine learning rate scheduling
* Threshold calibration using ROC curve optimization
* Streamlit Cloud deployment



## 🏥 Disclaimer

This project is intended for educational and research purposes only.
It is not a substitute for professional medical diagnosis.


