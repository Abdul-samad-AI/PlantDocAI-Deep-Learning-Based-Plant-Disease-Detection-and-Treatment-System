# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
from treatment_dict import treatment_dict  # Your detailed dictionary
import os
import pandas as pd
import datetime
from fpdf import FPDF


# ------------------------
# App Config
# ------------------------
st.set_page_config(
    page_title="PlantDocAI",
    page_icon="🌿",
    layout="wide"
)


# =======================
#  HEADER SECTION
# =======================
st.markdown("""
    <div style="background: rgba(31, 97, 65, 0.57); 
                padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white;">🌿 PlantDocAI</h1>
        <p style="color: white;">Detect plant diseases instantly & get expert treatment suggestions.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stExpander"] {
    border: 1px solid rgba(31, 97, 65, 0.3);
    border-radius: 8px;
    background-color: rgba(240,255,240,0.8);
}
div[data-testid="stExpander"] p {
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)




# ------------------------
# Set Background Image
# ------------------------
import base64

def set_bg_image(bg_image_path):
    with open(bg_image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    st.markdown("""
<style>
.stApp {
    font-family: 'Poppins', sans-serif;
}
h1, h2, h3 {
    color: #0BAB64;
}
div[data-testid="stFileUploader"] {
    background: rgba(31, 97, 65, 0.57);
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# Replace with your actual background image path
bg_image_path = r"C:\Users\LOQ\OneDrive\Documents\Work Space\PlantDocAI\background.jpg"
set_bg_image(bg_image_path)

# ------------------------
# Device & Model Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 38
classes = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot',
    'Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# Load model
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Linear(num_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(256, num_classes),
    torch.nn.LogSoftmax(dim=1)
)
model.load_state_dict(torch.load("best_resnet50_model.pth", map_location=device))
model = model.to(device)
model.eval()

predict_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------------------
# Prediction Function
# ------------------------
def predict_disease(image):
    img = image.convert("RGB")
    img_tensor = predict_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, pred = torch.max(probs, 1)
    class_name = classes[pred.item()]
    confidence = probs[0][pred.item()].item() * 100
    info = treatment_dict.get(class_name, "No treatment info available.")
    return class_name, confidence, info  

# ------------------------
# Convert results to CSV
# ------------------------
def convert_results_to_csv(results):
    df = pd.DataFrame(results)
    return df.to_csv(index=False).encode('utf-8')

# ------------------------
# Tabs: Single vs Batch
# ------------------------
tab1, tab2 = st.tabs(["📸 Single Image", "📁 Batch Prediction"])
all_results = []

# ------------------------
# Single Image Prediction
# ------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload a single image", type=["jpg", "jpeg", "png"], key="single_upload")
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        with col2:
            class_name, confidence, info = predict_disease(image)
            st.write(f"**Confidence:** {confidence:.2f}%")

            structured_info = {}
            for line in info.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    structured_info[key.strip()] = value.strip()
                else:
                    structured_info[line.strip()] = ""

            color = "green" if "healthy" in class_name.lower() else "red"
            st.markdown(f"<div style='border-left:5px solid {color}; padding:10px; border-radius:5px;'>", unsafe_allow_html=True)
            st.subheader(f"🪴 Predicted Disease: {class_name}")
            for key, value in structured_info.items():
                with st.expander(key):
                    st.write(value)
            if "Treatment" in structured_info:
                st.success(f"💊 Treatment: {structured_info['Treatment']}")
            if "Prevention" in structured_info:
                st.info(f"🧠 Prevention: {structured_info['Prevention']}")
            st.markdown("</div>", unsafe_allow_html=True)

            result_dict = {"image_name": uploaded_file.name, "class_name": class_name}
            result_dict.update(structured_info)
            all_results.append(result_dict)

# ------------------------
# Batch Prediction
# ------------------------
with tab2:
    folder_path = st.text_input("Enter folder path for images", key="batch_folder_input")
    if folder_path and os.path.exists(folder_path):
        images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        st.write(f"📸 Found **{len(images)}** images in the folder.")

        # Scrollable container for batch results
        with st.container():
            st.markdown("""
                <style>
                .scrollable-container {
                    max-height: 600px;
                    overflow-y: auto;
                    padding: 10px;
                    background: rgba(255, 255, 255, 0.7);
                    border-radius: 10px;
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown("<div class='scrollable-container'>", unsafe_allow_html=True)

            # Display images in grid layout
            cols = st.columns(3)
            for idx, img_name in enumerate(images):
                with cols[idx % 3]:
                    image_path = os.path.join(folder_path, img_name)
                    image = Image.open(image_path)
                    st.image(image, caption=img_name, use_container_width=True
)

                    # Prediction
                    class_name, confidence, info = predict_disease(image)

                    # Structure treatment info
                    structured_info = {}
                    for line in info.strip().split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            structured_info[key.strip()] = value.strip()
                        else:
                            structured_info[line.strip()] = ""

                    # Disease color tag
                    color = "green" if "healthy" in class_name.lower() else "red"
                    st.markdown(f"<div style='border-left:5px solid {color}; padding-left:10px; margin-top:8px; border-radius:5px;'>", unsafe_allow_html=True)
                    st.markdown(f"**🪴 Predicted Disease:** {class_name}")
                    st.markdown(f"**📊 Confidence:** {confidence:.2f}%")

                    # Scrollable details expander
                    with st.expander("🔍 View Detailed Report", expanded=False):
                        for key, value in structured_info.items():
                            if value.strip():
                                st.markdown(f"**{key}:** {value}")
                            else:
                                st.markdown(f"**{key}:** -")

                        if "Treatment" in structured_info:
                            st.success(f"💊 Treatment: {structured_info['Treatment']}")
                        if "Prevention" in structured_info:
                            st.info(f"🧠 Prevention: {structured_info['Prevention']}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Save result
                    result_dict = {"image_name": img_name, "class_name": class_name}
                    result_dict.update(structured_info)
                    all_results.append(result_dict)

            st.markdown("</div>", unsafe_allow_html=True)

# generate pdf
from fpdf import FPDF
import os

def generate_pdf(results, output_path="PlantDocAI_Report.pdf"):
    pdf = FPDF(format="A4")
    pdf.add_page()

    # Try to use a UTF-8 supporting font
    font_path = "DejaVuSans.ttf"
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=16)
    else:
        pdf.set_font("Helvetica", size=16)
        print(f"Warning: {font_path} not found. Using Helvetica. Unicode characters may not render correctly.")

    # Title
    pdf.cell(0, 10, "PlantDocAI Report", ln=True, align="C")

    # Set body font
    if os.path.exists(font_path):
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Helvetica", size=12)

    # Calculate safe content width
    page_width = pdf.w - 2 * pdf.l_margin

    # Write results
    for res in results:
        pdf.ln(8)
        pdf.set_x(pdf.l_margin)  # reset cursor to left margin before writing
        pdf.multi_cell(page_width, 10, f"Image: {res.get('image_name', 'N/A')}")

        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(page_width, 10, f"Disease: {res.get('class_name', 'N/A')}")

        for k, v in res.items():
            if k not in ["image_name", "class_name"]:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(page_width, 8, f"{k}: {v}")

        pdf.ln(4)

    pdf.output(output_path)
    return output_path


# ------------------------
# Download Results
# ------------------------
if all_results:
    st.markdown("---")
    st.markdown("### 📥 Download All Predictions")
    csv = convert_results_to_csv(all_results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="🧾Download CSV",
        data=csv,
        file_name=f"plant_disease_predictions_{timestamp}.csv",
        mime="text/csv"
    )

    pdf_path = generate_pdf(all_results)
    with open(pdf_path, "rb") as f:
       st.download_button(
        label="📄 Download PDF Report",
        data=f,
        file_name=f"plantdocai_report_{timestamp}.pdf",
        mime="application/pdf"
    )

st.markdown("""
<div style="background: rgba(31, 97, 65, 0.57); 
            padding: 15px; border-radius: 10px; text-align: center; color: white; margin-top: 30px;">
    <h4 style="margin:5px;">👨‍💻 Developed by <a href='https://www.linkedin.com/in/abdulsamad14' target='_blank' style='color:white; text-decoration:underline;'>ABDUL SAMAD</a></h4>
    <p style="margin:5px;">🌿 Model: ResNet50 &nbsp;&nbsp; | &nbsp;&nbsp; Dataset: PlantVillage &nbsp;&nbsp; | &nbsp;&nbsp; Version: 1.0</p>
    <p style="font-size:0.9em; margin:5px;">📌 Thank you for using PlantDocAI! 🌱</p>
</div>
""", unsafe_allow_html=True)
