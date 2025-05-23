import streamlit as st
from PIL import Image
import torch
from model import (
    load_image_model, predict_image_emotion, predict_text_sentiment, transform
)

# Cache model loading
@st.cache_resource
def get_model():
    return load_image_model()

model = get_model()

# Streamlit UI
st.title("Sentence Sentiment Predictor")

sentence = st.text_area("Enter a sentence to classify:", height=150)

if st.button("Predict Sentiment"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        result = predict_text_sentiment(sentence)
        st.success(f"Prediction: **{result}**")

st.title("Emotion Classifier (Facial Image)")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)
    label, _ = predict_image_emotion(model, input_tensor)
    st.markdown(f"### Predicted Emotion: `{label}`")
