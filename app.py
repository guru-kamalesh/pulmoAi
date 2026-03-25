import streamlit as st
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite


# ================= PAGE CONFIG =================
st.set_page_config(page_title="PulmoAI", layout="centered")


# ================= GLASS UI STYLE =================
st.markdown("""
<style>
.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}
.block-container{
background: rgba(255,255,255,0.08);
padding:30px;
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


st.markdown("<h1>🩺 PulmoAI</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI Lung Disease Detection from Chest X-ray</h3>", unsafe_allow_html=True)


# ================= LOAD TFLITE MODEL =================
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# ================= LOAD CLASS LABELS =================
with open("class_indices.json") as f:
    class_indices = json.load(f)

classes = list(class_indices.keys())


# ================= XRAY VALIDATION =================
def is_xray(img):
    img = img.resize((224,224))
    arr = np.array(img)

    gray = np.mean(np.abs(arr[:,:,0] - arr[:,:,1])) + \
           np.mean(np.abs(arr[:,:,1] - arr[:,:,2]))

    return gray < 15


# ================= MEDICAL SUMMARY =================
def medical_summary(label):

    if label == "COVID":
        return "Radiographic patterns suggest viral infection consistent with COVID-19 with possible ground-glass opacities."

    elif label == "LUNG_OPACITY":
        return "Localized lung opacity detected which may indicate inflammation, consolidation or fluid accumulation."

    elif label == "VIRAL_PNEUMONIA":
        return "Diffuse interstitial infiltrates indicate viral pneumonia affecting lung parenchyma."

    else:
        return "No significant abnormal radiographic findings detected. Lung fields appear clear."


# ================= FILE UPLOAD =================
file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg","png","jpeg"])


if file:

    img = Image.open(file).convert("RGB")

    # validate xray
    if not is_xray(img):
        st.error("⚠️ Uploaded image does not appear to be a Chest X-ray.")
        st.stop()

    st.image(img, caption="Uploaded Radiograph", use_column_width=True)

    # preprocess
    img_resized = img.resize((224,224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype("float32")

    # predict
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_class = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.markdown("---")

    # result card
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


    # ================= GRAPH =================
    st.markdown("### 📊 Disease Probability Analysis")

    fig, ax = plt.subplots(figsize=(8,4))

    bars = ax.bar(classes, pred * 100)
    bars[np.argmax(pred)].set_color("green")

    ax.set_ylim(0,100)
    ax.set_ylabel("Probability (%)")
    ax.set_title("AI Diagnostic Confidence")
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2,
                height + 1,
                f'{height:.1f}%',
                ha='center')

    st.pyplot(fig)


    # ================= MEDICAL SUMMARY =================
    st.markdown("### 🧠 Medical Interpretation")
    st.info(medical_summary(predicted_class))


    st.markdown("---")
    st.caption("⚠️ This AI system is an assistive diagnostic tool and not a replacement for professional medical diagnosis.")
