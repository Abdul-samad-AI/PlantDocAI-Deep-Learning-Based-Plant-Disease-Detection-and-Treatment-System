# ==========================================================
# 🌿 PlantDocAI — Intelligent Plant Disease Detection System
# ==========================================================

# ----------------------------------------------------------
# 📌 Overview
# ----------------------------------------------------------
# PlantDocAI is an end-to-end AI-powered web application
# that detects plant diseases from leaf images and provides
# actionable treatment recommendations.
#
# The system is designed for:
# - Farmers
# - Agronomists
# - Agricultural researchers
# - Agri-tech platforms
#
# It delivers real-world diagnostic insights including:
# - Disease identification
# - Confidence score
# - Cause & symptoms
# - Treatment recommendations
# - Suggested commercial products
# - Preventive measures


# ----------------------------------------------------------
# 🚀 Key Highlights
# ----------------------------------------------------------
# - Deep Learning-based disease classification
# - Covers 38 plant disease & healthy classes
# - High-accuracy ResNet50 Transfer Learning model
# - Interactive Streamlit web application
# - Single image & batch prediction support
# - CSV and professional PDF report generation
# - Modular, scalable, and production-ready design


# ==========================================================
# 🧠 Model Overview
# ==========================================================

Architecture:        ResNet50 (Convolutional Neural Network)
Training Strategy:   Transfer Learning (ImageNet pretrained)
Loss Function:       Cross-Entropy Loss
Optimizer:           Adam
Input Size:          224 x 224 RGB images
Number of Classes:   38 (Healthy + Diseased plant categories)
Dataset:             PlantVillage (curated & preprocessed)


# ==========================================================
# 🖥️ Application Features
# ==========================================================

# ----------------------------------------------------------
# 🔹 Single Image Prediction
# ----------------------------------------------------------
# - Upload a plant leaf image
# - Get instant prediction with:
#     • Disease name
#     • Confidence score (%)
# - Expandable diagnostic sections:
#     • Cause
#     • Symptoms
#     • Treatment
#     • Recommended products
#     • Prevention steps
# - Color-coded results:
#     • Green → Healthy
#     • Red → Diseased


# ----------------------------------------------------------
# 🔹 Batch Prediction
# ----------------------------------------------------------
# - Predict multiple images from a folder
# - Visual grid-based image display
# - Per-image disease classification
# - Suitable for farm-level or field analysis


# ----------------------------------------------------------
# 📥 Export & Reporting
# ----------------------------------------------------------
# - Download CSV reports with timestamps
# - Generate professional PDF diagnostic reports
# - Designed for real-world agricultural documentation
# - Reports suitable for sharing with agronomists or clients


# ==========================================================
# 🏗️ Project Structure
# ==========================================================
The dataset is intentionally **not included** in this repository due to size constraints.
```
PlantDocAI/
│
├── app.py                      # Main Streamlit application
├── treatment_dict.py           # Disease → Treatment mapping
├── best_resnet50_model.pth     # Trained model weights
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
│
├── data/                       # Not Uploaded
│   ├── train/                  # Training dataset
│   ├── valid/                  # Validation dataset
│   └── test/                   # Test images (optional)
│
├── assets/
│   └── background.jpg          # UI background image
│
└── reports/
    ├── sample_report.pdf       # Example generated PDF
    └── predictions.csv         # Example CSV output
```

# ==========================================================
# ⚙️ Installation & Setup
# ==========================================================

# 1. Clone the repository
git clone https://github.com/your-username/PlantDocAI.git
cd PlantDocAI

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py


# ==========================================================
# 📊 Dataset Information
# ==========================================================

# Dataset Name: PlantVillage
# Total Classes: 38
# Image Type: RGB leaf images
# Preprocessing:
# - Resize to 224 x 224
# - Normalization using ImageNet statistics
# - Data augmentation during training


# ==========================================================
# 🧪 Model Performance
# ==========================================================

# - Transfer learning improves convergence speed
# - High validation accuracy achieved on PlantVillage
# - Robust generalization on unseen leaf images
# - Confidence score provided for transparency

# ==========================================================
# 📸 Screenshots
# ==========================================================


# ==========================================================
# 🔮 Future Improvements
# ==========================================================

# - Mobile-friendly deployment (Android / iOS)
# - Real-time camera-based disease detection
# - Multilingual support for farmers
# - Weather-aware disease prediction
# - Cloud deployment (AWS / GCP / Azure)
# - Integration with agricultural advisory APIs


# ==========================================================
# 👨‍💻 Author & Credits
# ==========================================================

Developed by:        Abdul Samad
LinkedIn:            https://www.linkedin.com/in/abdulsamad14
Model:               ResNet50
Dataset:             PlantVillage
Version:             1.0

# ----------------------------------------------------------
# 🌱 Thank you for using PlantDocAI!
# ----------------------------------------------------------
