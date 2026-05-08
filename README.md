HelioSight AI ☀️

Enterprise-style AI-powered thermal solar panel inspection platform using Deep Learning, Computer Vision, and Streamlit.

🚀 Overview

HelioSight AI is a drone-based solar inspection analytics platform that detects thermal defects in solar panels using infrared imagery and deep learning.

The platform performs:

Real-time thermal image inspection
AI-based defect classification
Solar infrastructure monitoring
Inspection analytics & reporting
Renewable energy defect analysis

Built using TensorFlow, OpenCV, Streamlit, and MobileNetV2 transfer learning.

📌 Features
🔥 Thermal solar panel defect detection
🤖 CNN-based deep learning inference
📊 Interactive analytics dashboard
📁 Real-time image upload & inspection
📈 Defect probability visualization
🧠 MobileNetV2 transfer learning pipeline
📄 Downloadable inspection reports
🌙 Enterprise-grade dark UI
🧠 AI Model
Property	Value
Architecture	MobileNetV2
Framework	TensorFlow / Keras
Dataset	Infrared Solar Modules Dataset
Images Used	20,000+
Classes	12
Accuracy	62.5%
Input Size	128x128
🔍 Defect Classes
No-Anomaly
Cell
Cell-Multi
Cracking
Diode
Diode-Multi
Hot-Spot
Hot-Spot-Multi
Offline-Module
Shadowing
Soiling
Vegetation
🛠️ Tech Stack
Frontend
Streamlit
HTML/CSS
Plotly
Backend / AI
Python
TensorFlow
Keras
OpenCV
NumPy
Pandas
Scikit-learn
📂 Project Structure
HelioSight-AI/
│
├── app/
│   ├── assets/
│   ├── components/
│   ├── pages/
│   └── utils/
│       └── predictor.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── cnn_model.h5
│   ├── label_encoder.pkl
│   └── model_meta.json
│
├── scripts/
│   ├── main.py
│   ├── preprocess_images.py
│   └── train_model.py
│
├── requirements.txt
├── config.py
└── .gitignore
⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/HelioSight-AI.git
cd HelioSight-AI
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Application
streamlit run scripts/main.py
🧪 Model Training
Preprocess Dataset
python scripts/preprocess_images.py
Train CNN Model
python scripts/train_model.py
📊 Dashboard Modules
Dashboard Analytics
Thermal Inspection
Model Insights
Inspection Reports
🌍 Real-World Use Cases
Solar farm inspection
Drone-based thermal monitoring
Renewable energy analytics
Predictive maintenance
Industrial AI inspection systems
👩‍💻 Developer

Sejal Rane
AI/ML Undergraduate | Computer Vision & Analytics Enthusiast

⭐ Future Improvements
YOLO-based defect localization
Live drone feed integration
Real-time thermal video inference
Cloud deployment
Multi-user inspection system
📜 License

This project is for educational and portfolio purposes.
