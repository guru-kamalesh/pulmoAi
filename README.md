# 🩺 PulmoAI – AI Lung Disease Detection System

PulmoAI is a Deep Learning based medical imaging web application that detects lung diseases from Chest X-ray images.

The system performs multi-class classification and provides an AI-assisted medical interpretation with confidence visualization.

------------------------------------------------------------

🚀 Problem Statement

Early detection of respiratory diseases such as COVID-19 and Pneumonia is crucial but requires expert radiological analysis.

PulmoAI aims to support healthcare professionals by providing:

• Fast AI-based preliminary screening  
• Multi-disease classification  
• Confidence-driven diagnostic insights  
• User-friendly medical interface  

------------------------------------------------------------

🧠 Features

✅ Multi-Disease Detection  
• COVID-19  
• Viral Pneumonia  
• Lung Opacity  
• Normal  

✅ Smart X-ray Validation  
Rejects non-radiography images automatically.

✅ Medical Interpretation  
Displays clinically styled diagnostic summary.

✅ Confidence Visualization  
Professional probability analysis graph.

✅ Modern Glassmorphism UI  
Responsive healthcare themed web interface.

------------------------------------------------------------

🏗️ System Workflow

User Uploads Chest X-ray  
↓  
Image Preprocessing & Normalization  
↓  
Deep Learning Model (MobileNetV2 Transfer Learning)  
↓  
Multi-Class Prediction  
↓  
Confidence Analysis + Medical Summary  
↓  
Interactive Streamlit Web Interface  

------------------------------------------------------------

⚙️ Technology Stack

• Python  
• TensorFlow / Keras  
• MobileNetV2  
• Streamlit  
• NumPy  
• Matplotlib  

------------------------------------------------------------

📊 Model Details

• Input Image Size : 224 × 224  
• Architecture : Transfer Learning  
• Output Layer : Softmax Multi-Class Classification  
• Training Strategy : Frozen Base Layers  

------------------------------------------------------------

💻 How to Run Locally

Step 1 — Install Dependencies

pip install -r requirements.txt

Step 2 — Run Application

streamlit run app.py

------------------------------------------------------------

📁 Project Structure

PulmoAI/
│
├── app.py  
├── main.py  
├── model/  
│   ├── model.h5  
│   └── class_indices.json  
├── requirements.txt  
├── screenshots/  
└── README.md  

------------------------------------------------------------

📸 Screenshots

(Add UI screenshots here for better presentation)

------------------------------------------------------------

⚠️ Disclaimer

This AI system is intended for assistive diagnostic purposes only  
and should not replace professional medical evaluation.

------------------------------------------------------------

🌟 Future Enhancements

• Lung infection heatmap visualization  
• Automatic PDF medical report generation  
• Model optimization & higher accuracy  
• Cloud deployment  
• Multi-modal disease analysis  

------------------------------------------------------------

👨‍💻 Author
Guru
Deepak
Akshay
Sujay
AI / ML Hackathon Project  

------------------------------------------------------------

⭐ If you find this project useful, consider giving it a star on GitHub.
