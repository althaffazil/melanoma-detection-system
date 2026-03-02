# Melanoma Detection and Lesion Segmentation using Deep Learning

An end-to-end deep learning system for automated skin lesion segmentation and melanoma classification, containerized using Docker for portable and reproducible deployment.

This project integrates medical image segmentation (U-Net) and melanoma classification (EfficientNet) into a modular, production-ready application.


## 🚀 Deployment Options

### 🔹 Run via Docker Hub (Prebuilt Image)

Pull the CPU-based Docker image:

```bash
docker pull cybervamp/melanoma-detection-cpu
````

Run the container:

```bash
docker run -p 8501:8501 cybervamp/melanoma-detection-cpu
```

Access the application at:

```
http://localhost:8501
```


### 🔹 Build Locally with Docker

Clone the repository:

```bash
git clone https://github.com/althaffazil/melanoma-detection-system.git
cd melanoma-detection-system
```

Build the Docker image:

```bash
docker build -t melanoma-detection -f docker/Dockerfile .
```

Run the container:

```bash
docker run -p 8501:8501 melanoma-detection
```



## 🩺 Project Overview

This system performs two core tasks:

### 1️⃣ Lesion Segmentation

* Model: U-Net
* Task: Pixel-wise lesion mask prediction
* Dice Score: ~0.92
* IoU: ~0.85

### 2️⃣ Melanoma Classification

* Backbone: EfficientNet
* Metric: ROC-AUC (~0.93–0.96)
* Class imbalance handled via `pos_weight`
* Decision threshold optimized using ROC analysis (Youden’s J statistic)

The deployed application allows users to:

* Upload dermoscopic images
* Visualize lesion segmentation overlay
* View melanoma probability score
* Receive model-based diagnostic decision



## 🏗️ System Architecture

### 🔹 Segmentation Module

* U-Net architecture
* BCE + Dice-based training
* Binary mask prediction
* Post-processing thresholding for clean overlays

### 🔹 Classification Module

* EfficientNet backbone
* Binary melanoma classification
* Stratified 80/20 train-validation split
* Mixed precision training (during development)
* Cosine learning rate scheduler
* Threshold calibrated via ROC curve optimization



## 📂 Folder Structure

```
.
├── app/
│   └── streamlit_app.py        # Streamlit entry point
│
├── checkpoints/
│   ├── classifier_best.pth     # Best trained classification model
│   └── segmentation_best.pth   # Best trained segmentation model
│
├── docker/
│   └── Dockerfile              # Docker build configuration
│
├── src/
│   ├── datasets/
│   │   ├── classification_dataset.py
│   │   └── segmentation_dataset.py
│   │
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
├── .dockerignore
└── README.md
```


## ⚙️ Local (Non-Docker) Installation

```bash
git clone https://github.com/althaffazil/melanoma-detection-system.git
cd melanoma-detection-system
pip install -r requirements.txt
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
* Mixed precision training for GPU efficiency (development phase)
* Cosine learning rate scheduling
* Threshold calibration using ROC curve optimization
* Containerized deployment using Docker
* Portable CPU-based production image via Docker Hub



## 🏥 Disclaimer

This project is intended for educational and research purposes only.
It is not a substitute for professional medical diagnosis.
