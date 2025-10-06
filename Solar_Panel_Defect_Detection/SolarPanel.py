import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# -------------------------
# Set your Google Drive model paths
# -------------------------
MODEL_PATHS = {
    "ResNet18": "C:/Project/AIML/Solar_Panel_Defect_Detection/Model/resnet18_solarguard.pth",
    "MobileNetV2": "C:/Project/AIML/Solar_Panel_Defect_Detection/Model/mobilenet_v2_solarguard.pth",
    "EfficientNetB0": "C:/Project/AIML/Solar_Panel_Defect_Detection/Model/efficientnet_b0_solarguard.pth"
}

CLASS_NAMES = ["Clean", "Dusty", "Bird-Drop", "Electrical-Damage", "Physical-Damage", "Snow-Covered"]



# Load model architecture & weights
@st.cache_resource
def load_model(name, path):
    if name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    elif name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    elif name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    else:
        raise ValueError("Unknown model name")

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image


# Predict
def get_prediction(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item()
    return CLASS_NAMES[predicted_class], confidence


st.set_page_config(page_title="Solar Panel", layout="centered")
st.title("SolarGuard: Intelligent Defect Detection on Solar Panels")
st.markdown("Upload a solar panel image to classify its condition using 3 deep learning models.")

uploaded_file = st.file_uploader("Upload an image of a solar panel", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_tensor = preprocess_image(image)

    results = []

    for model_name, model_path in MODEL_PATHS.items():
        try:
            model = load_model(model_name, model_path)
            prediction, confidence = get_prediction(model, image_tensor)
            results.append((model_name, prediction, confidence))
        except Exception as e:
            results.append((model_name, f" Error loading model: {e}", 0.0))

    st.subheader("Model Predictions")
    for name, pred, conf in results:
        st.write(f"**{name}** → `{pred}` with **{conf*100:.2f}%** confidence")

    # Recommendation
    if results:
        common_preds = [r[1] for r in results]

        # Define priority order: defects first
        defect_priority = ["Dusty", "Snow-Covered"]
        repair_priority = ["Bird-Drop","Electrical-Damage","Physical-Damage"]
        
        # Check if any defect is present in predictions
        detected_defect = next((d for d in defect_priority if d in common_preds), None)
        detected_repair = next((d for d in repair_priority if d in common_preds), None)


        st.subheader("Recommendation")

        if detected_defect:
            st.warning(f"Detected obstruction: **{detected_defect}**. Schedule cleaning.")
        elif detected_repair:
            st.warning(f"Detected possible damage: **{detected_repair}**. Recommend inspection/repair.")
        elif "Clean" in common_preds:
            st.success("Panel is clean. No action needed.")
        

