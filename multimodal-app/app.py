import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
import joblib
import numpy as np
from model import MultimodalClassifier

# Emotion Classes
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load TF-IDF vectorizer and label encoder
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load model
tfidf_dim = len(tfidf_vectorizer.get_feature_names_out())
num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultimodalClassifier(tfidf_dim=tfidf_dim, num_classes=num_classes)
model.load_state_dict(torch.load("fusion_emotion_best.pth", map_location=device))
model.eval().to(device)

# Image transform
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("Multimodal Emotion Classifier")
text_input = st.text_area("Enter dialogue utterance:")
image_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if image_file is not None:
    image = Image.open(image_file).convert("RGB")
    st.image(image)

if st.button("Classify") and text_input and image_file:
    # Process text
    tfidf_vec = tfidf_vectorizer.transform([text_input]).toarray()
    tfidf_tensor = torch.tensor(tfidf_vec, dtype=torch.float32).to(device)

    # Process image
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logits = model(tfidf_tensor, image_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        label = EMOTION_CLASSES[pred_class]
        st.success(f"Predicted Emotion: **{label}**")
