import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords (ensure it's run once)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Emotion classes for image model
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Load image model
def load_image_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load("resnet_emotion_best.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

# Predict emotion from image tensor
def predict_image_emotion(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred_class = EMOTION_CLASSES[torch.argmax(probs)]
    return pred_class, probs.numpy()[0]

# Text preprocessor
def preprocessor(sentence):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', sentence).lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return ' '.join(review)

# Label mapper
def mapper(ans):
    label_map = {
        0: 'Anger', 1: 'Disgust', 2: 'Fear',
        3: 'Joy', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'
    }
    return label_map[ans[0]]

# Predict sentiment from sentence
def predict_text_sentiment(sentence):
    tfv = joblib.load('tfidf.pkl')
    my_model = joblib.load('lr2.pkl')
    processed = preprocessor(sentence)
    vector = tfv.transform([processed])
    prediction = my_model.predict(vector)
    return mapper(prediction)
