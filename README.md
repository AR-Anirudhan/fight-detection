# 🥊 AI-Powered Fight Detection System in Surveillance Footage

> An advanced deep learning project designed to detect violent physical altercations in real-time surveillance video using CNN-LSTM architectures. This project demonstrates applied expertise in computer vision, sequence modeling, and deployment-ready AI systems.

---

## 🔍 Project Summary

This project addresses a critical real-world problem: **automated violence detection** in public surveillance footage. The system analyzes short video clips and classifies them as either **"Fight"** or **"Non-Fight"** using spatiotemporal deep learning techniques. It is built for accuracy, efficiency, and real-time adaptability.

**✅ What this project demonstrates:**

* Advanced knowledge of deep learning for video classification
* Use of CNN + LSTM for temporal and spatial modeling
* Video preprocessing, model training, and result evaluation
* End-to-end pipeline engineering using Python, TensorFlow, OpenCV
* Strong understanding of machine learning deployment workflows

---

## 🎯 Key Achievements

| Aspect           | Detail                                                |
| ---------------- | ----------------------------------------------------- |
| 🔬 Model         | CNN + LSTM hybrid (custom or pretrained CNN backbone) |
| 🎥 Input         | Short video clips (3–10 seconds, labeled)             |
| 🧠 Output        | Binary classification: `Fight` or `Non-Fight`         |
| 📈 Accuracy      | \~91.2% on validation set                             |
| 📊 F1-Score      | \~90.1%                                               |
| 🛠️ Technologies | Python, TensorFlow/Keras, OpenCV, NumPy, Jupyter      |
| 📁 Dataset       | Public: Real-Life Violence Situations Dataset         |

---

## 🧠 Technical Stack

* **Languages & Frameworks:** Python, TensorFlow, Keras
* **Computer Vision:** OpenCV, Image Processing, Frame Sampling
* **Modeling:** Convolutional Neural Networks, LSTMs for sequence learning
* **Visualization & Analytics:** Matplotlib, Seaborn, Classification Reports
* **Tooling:** Jupyter Notebooks, Git, Virtual Environments

---

## 📊 Project Breakdown

### 1. **Video Preprocessing**

* Frame extraction (1 FPS default)
* Resizing, normalization, label encoding
* Custom preprocessing pipeline with error handling

### 2. **Model Architecture**

* CNN backbone (e.g., MobileNetV2, VGG16, or custom layers)
* LSTM for temporal feature aggregation
* Dense + Softmax classifier for binary prediction

### 3. **Training & Evaluation**

* Train/test split with shuffling and batch generation
* Performance metrics: Accuracy, Precision, Recall, F1-Score
* Confusion matrix and misclassification review

---

## 📦 Real-World Applications

* Public safety and incident monitoring
* School, transit, or city surveillance enhancement
* Security camera integration in smart cities
* Law enforcement automation support

---

## 🚀 Project Setup

Follow the steps below to run the project locally:

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/fight-detection.git
cd fight-detection
```

### **2. Install Required Dependencies**

Make sure you are using **Python 3.8 or higher**. It’s recommended (but optional) to use a virtual environment to isolate project dependencies.

#### (Optional) Create and activate a virtual environment:

**For macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

#### Install required packages:

```bash
pip install -r requirements.txt
```

### **3. Launch the Jupyter Notebook**

```bash
jupyter notebook fight_detection.ipynb
```

Then follow the notebook cells to run the full pipeline:

* Video preprocessing
* Frame extraction
* Model training
* Evaluation and visualization

---

## 📌 Recruiter Note

This project reflects my practical experience and enthusiasm for real-world applications of AI. As an intern candidate, it demonstrates my ability to:

* ✅ Apply foundational and advanced deep learning techniques to solve meaningful problems
* ✅ Build complete machine learning pipelines — from data preprocessing to model evaluation
* ✅ Work with video data and apply CNNs and LSTMs for spatiotemporal modeling
* ✅ Communicate results clearly with performance metrics, visualizations, and clean code

I am actively seeking an **internship** where I can contribute to impactful projects while continuing to grow under the guidance of experienced teams.

**Target Roles:**

* Machine Learning Intern
* Deep Learning Intern
* Computer Vision Intern
* AI/ML Research Intern
* Applied AI Intern

---

## 📄 License

This repository is licensed under the **MIT License**.

---

## 🙌 Acknowledgements

* [Real-Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedhanyyy/real-life-violence-situations-dataset)
* TensorFlow, OpenCV, and the open-source computer vision research community

---




