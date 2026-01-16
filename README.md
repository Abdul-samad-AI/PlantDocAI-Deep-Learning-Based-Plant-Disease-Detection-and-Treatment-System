---

# 🌿 PlantDocAI

### Deep Learning–Based Plant Disease Detection & Treatment Recommendation System

PlantDocAI is an end-to-end **AI-powered web application** that detects plant diseases from leaf images and provides **actionable treatment recommendations**, including **causes, symptoms, prevention strategies, and suggested products**.

The project demonstrates the **practical application of Deep Learning in Agriculture** and is designed for **farmers, agronomists, researchers, and agri-tech platforms**.

---

## 🚀 Key Features

* 🌱 Detects **38 plant disease & healthy classes**
* 🧠 High-accuracy **ResNet50 Transfer Learning model**
* 📸 Single image & batch image prediction
* 📊 Confidence score for every prediction
* 💊 Detailed treatment & prevention guidance
* 📁 CSV & professional PDF report generation
* 🖥️ Interactive **Streamlit web interface**
* 🎨 Clean, modern, and user-friendly UI

---

## 🧠 Model Overview

| Component         | Details                                 |
| ----------------- | --------------------------------------- |
| Architecture      | ResNet50 (CNN)                          |
| Training Strategy | Transfer Learning (ImageNet pretrained) |
| Loss Function     | Cross-Entropy Loss                      |
| Optimizer         | Adam                                    |
| Input Size        | 224 × 224 RGB images                    |
| Classes           | 38 (Healthy + Diseased)                 |
| Dataset           | PlantVillage                            |

---

## 🖥️ Application Functionality

### 🔹 Single Image Prediction

* Upload a plant leaf image
* Get:

  * Disease name
  * Confidence score (%)
  * Expandable details:

    * Cause
    * Symptoms
    * Treatment
    * Recommended products
    * Prevention steps
* Color-coded output:

  * 🟢 Healthy
  * 🔴 Diseased

### 🔹 Batch Prediction

* Predict multiple images at once
* Visual grid-based results
* Suitable for farm-level or field-level analysis

### 📥 Export & Reports

* Download **CSV reports** with timestamps
* Generate **professional PDF diagnostic reports**
* Useful for documentation, research, and advisory use

---

## 🏗️ Project Structure

```
PlantDocAI/
│
├── app.py                     # Main Streamlit application
├── treatment_dict.py          # Disease → Treatment mapping
├── best_resnet50_model.pth    # Trained model weights (not tracked)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .gitignore                 # Git ignore rules
│
├── assets/
│   └── background.jpg         # UI background image
│
├── data/                      # Dataset (not included in repo)
│   ├── train/
│   ├── valid/
│   └── test/
│
└── reports/
    ├── sample_report.pdf
    └── predictions.csv
```

---

## 📦 Dataset Information

* **Dataset Name:** PlantVillage
* **Total Classes:** 38
* **Dataset Size:** ~5 GB
* **Image Type:** RGB leaf images

📥 Dataset download:
[https://www.kaggle.com/datasets/emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease)

> ⚠️ The dataset is **not included in this repository** due to size constraints.
> After downloading, place it inside the `data/` folder as shown above.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Abdul-samad-AI/PlantDocAI-Deep-Learning-Based-Plant-Disease-Detection-and-Treatment-System.git
cd PlantDocAI-Deep-Learning-Based-Plant-Disease-Detection-and-Treatment-System
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

Activate:

* Windows:

```bash
venv\Scripts\activate
```

* Linux / macOS:

```bash
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
streamlit run app.py
```

---

## 📊 Model Performance

* High validation accuracy on PlantVillage dataset
* Robust predictions on unseen images
* Confidence score improves transparency & trust
* Transfer learning reduces training time significantly

---

## 🔮 Future Enhancements

* 📱 Mobile application support
* 🌍 Multilingual farmer-friendly interface
* 🌦️ Weather-aware disease prediction
* ☁️ Cloud deployment (AWS / GCP)
* 📷 Real-time camera-based detection
* 🧪 Severity estimation & yield loss analysis

---

## 👨‍💻 Author

**Abdul Samad**
🔗 LinkedIn: [https://www.linkedin.com/in/abdulsamad14](https://www.linkedin.com/in/abdulsamad14)

**Tech Stack:** Python, PyTorch, Streamlit
**Model:** ResNet50
**Dataset:** PlantVillage
**Version:** 1.0

---

## 🌱 Acknowledgements

* PlantVillage Dataset
* PyTorch & Torchvision
* Streamlit Community

---

⭐ If you find this project useful, consider giving it a **star** on GitHub.

---

If you want, I can also:

* Add **GitHub badges**
* Optimize this for **resume bullets**
* Prepare a **deployment README**
* Write a **research-paper style abstract**

Just tell me.
