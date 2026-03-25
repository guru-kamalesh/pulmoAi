import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt

# ================= UI CONFIG =================
st.set_page_config(page_title="AI Lung Diagnosis", layout="centered")

st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.block-container{
background: rgba(255,255,255,0.08);
padding:35px;
border-radius:20px;
backdrop-filter: blur(15px);
box-shadow:0 8px 32px rgba(0,0,0,0.4);
}

h1,h2,h3{
text-align:center;
color:white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🩺 AI Lung Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3>Deep Learning Chest Radiography Analysis</h3>", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = tf.keras.models.load_model("model/model.h5")

with open("model/class_indices.json") as f:
    class_indices = json.load(f)

classes = list(class_indices.keys())

# ================= XRAY VALIDATION =================
def is_xray(img):
    img = img.resize((224,224))
    arr = np.array(img)

    color_var = np.std(arr[:,:,0]-arr[:,:,1]) + np.std(arr[:,:,1]-arr[:,:,2])
    gray = np.mean(np.abs(arr[:,:,0]-arr[:,:,1])) + np.mean(np.abs(arr[:,:,1]-arr[:,:,2]))

    if color_var < 15 and gray < 15:
        return True
    return False

# ================= MEDICAL SUMMARY =================
def medical_summary(label):
    if label == "COVID":
        return "Radiographic features suggest viral infection consistent with COVID-19 showing ground-glass opacities."
    elif label == "LUNG_OPACITY":
        return "Localized lung opacity detected indicating inflammation, consolidation or fluid accumulation."
    elif label == "VIRAL_PNEUMONIA":
        return "Diffuse interstitial infiltrates indicate viral pneumonia affecting lung tissue."
    else:
        return "No significant abnormal radiographic findings. Lung fields appear clear."

# ================= FILE UPLOAD =================
file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")

    if not is_xray(img):
        st.error("⚠️ Uploaded image does not appear to be a valid Chest X-ray.")
        st.stop()

    st.image(img, caption="Uploaded Radiograph", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)[0]
    predicted_class = classes[np.argmax(pred)]
    confidence = np.max(pred)*100

    st.markdown("---")

    # RESULT CARD
    st.markdown(f"""
    <div style="
    background:rgba(0,255,150,0.15);
    padding:25px;
    border-radius:15px;
    text-align:center;
    font-size:24px;">
    🧾 Diagnosis : <b>{predicted_class}</b><br>
    🎯 Confidence : {confidence:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # ================= PROFESSIONAL GRAPH =================
    st.markdown("### 📊 Disease Probability Analysis")

    fig, ax = plt.subplots(figsize=(8,4))

    bars = ax.bar(classes, pred*100)

    # highlight predicted
    bars[np.argmax(pred)].set_color("green")

    ax.set_ylabel("Probability (%)")
    ax.set_xlabel("Disease Class")
    ax.set_title("AI Diagnostic Confidence")
    ax.set_ylim(0,100)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,
                height+1,
                f'{height:.1f}%',
                ha='center')

    st.pyplot(fig)

    # ================= MEDICAL INTERPRETATION =================
    st.markdown("### 🧠 Medical Interpretation")
    st.info(medical_summary(predicted_class))

    st.markdown("---")
    st.caption("⚠️ AI system is an assistive diagnostic tool — not a replacement for professional medical advice.")