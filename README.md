# EmotionScope
Multimodal Emotion Recognition in Conversation through Image and Text Fusion

IEEE Envision Project 2025

### Mentors
- Vanshika Mittal
- Rakshith Ashok Kumar
### Mentees
- Pradyun Diwakar
- Shriya Bharadwaj
- Vashisth Patel
- Deepthi Komar
- Karthikeya Gupta
- Kowndinya Vasudev
  
## Aim
To develop a multimodal system using the MELD dataset by integrating textual and visual inputs through a late fusion architecture, performing emotion classification in conversational settings.
## Introduction and Overview
Emotions are key to effective communication, influencing interactions and decision-making. This project aims to bridge the gap between humans and machines by recognizing emotions in conversations using both text and facial expressions. Leveraging the MELD dataset, we implement two parallel modules: 
- a TF-IDF-based NLP model for dialogue processing
- a ResNet-18-based vision model for facial expression analysis. 
- By combining these through a late fusion strategy, our system achieves more accurate emotion detection.
  
![es-1](https://github.com/user-attachments/assets/159f20c2-f38b-4668-b052-f94eb4daa7d7)

## Technologies Used
1. Python
2. PyTorch
3. Streamlit

## Dataset
The [MELD (Multimodal EmotionLines Dataset)](https://affective-meld.github.io/) is a benchmark dataset for emotion recognition in multi-party conversations, derived from the TV show Friends. It contains over 13,000 utterances across 1,400+ dialogues, each labeled with one of seven emotions: anger, disgust, fear, joy, neutral, sadness, or surprise. Each utterance is paired with text, audio, and video, enabling multimodal analysis. MELD retains conversational context and speaker information, making it ideal for studying emotion dynamics in dialogue. 

In our project, we use its text and visual components to build a multimodal emotion classification system.
#### Visualisation of emotion distribution in MELD:
![es-7](https://github.com/user-attachments/assets/6590d5ce-67ca-4935-a8e1-4b01fc108fbd)

## Model and Architecture
### 1. Textual Feature Extraction
In our project, we utilize the **TF-IDF (Term Frequency-Inverse Document Frequency)** representation in combination with **Logistic Regression** to classify the emotional content of dialogue utterances in the MELD dataset. 

TF-IDF is a numerical statistic that reflects how important a word is to a document in a collection (or corpus). It balances two components:
  - _Term Frequency (TF)_: Measures how frequently a term appears in a document. A higher frequency indicates greater importance.
  - _Inverse Document Frequency (IDF)_: Measures how unique or rare a term is across all documents. Rare terms across the corpus receive higher weights.

Logistic Regression is a linear classifier that we used to model the probability that a given input belongs to a particular class using the softmax function.

### 2. Visual Feature Extraction from Images
The presence of images alongside text helps in capturing subtle emotional cues more effectively during emotion analysis. To make use of this, we extracted keyframes from videos present in the MELD dataset — selecting frames where different individuals displayed distinct emotions. These facial images were then mapped to the corresponding utterances and labelled emotions from the textual dataset. 

To extract meaningful visual features from each face, we used a **Residual Neural Network (ResNet-18)**. This architecture is composed of stacked _3×3 convolutional layers_, each followed by _batch normalization_ and _ReLU activation_. Towards the end, an _adaptive average pooling layer_ reduces the spatial dimensions of the feature maps to a fixed size, enabling consistent output regardless of input image size. Since our task involved predicting 7 emotion classes (instead of the 1000 classes used in ImageNet), we removed the final fully connected (classification) layer of ResNet-18.

![es-2](https://github.com/user-attachments/assets/fa135af1-e0f5-4707-b078-8c63dc7df4bb)


### 3. Multimodal Fusion
Our emotion recognition model employs **late fusion** to integrate insights from textual and visual data using the MELD dataset. Text features are extracted using TF-IDF, followed by classification through Logistic Regression, capturing linguistic indicators of emotion. Visual cues, such as facial expressions, are processed using a ResNet architecture, which effectively extracts deep spatial features from images. This modular approach ensures that the strengths of each modality are preserved without interference during feature learning, handles noisy or missing data better, and avoids the complexities of early fusion. 

![es-8](https://github.com/user-attachments/assets/89deb848-6443-4e83-8c52-79e85213add0)

The result is a more accurate and context-aware emotion recognition system leveraging complementary cues from both language and facial expressions.

## Results
To evaluate the performance of our emotion recognition system, we conducted experiments across three setups: 
### 1. Text-only classification using TF-IDF + Logistic Regression
The TF-IDF-based model performed reasonably well on shorter utterances and common emotion categories. However, it struggled with context-dependent emotions such as sarcasm.

### 2. Image-only classification using ResNet-18
Using ResNet-18 for facial expression classification offered moderate performance. The model was sensitive to facial visibility, lighting, and resolution—limitations inherent to static frame analysis.

### 3. Multimodal classification using a late fusion of both models
By combining predictions from both modalities using a late fusion strategy, we observed a significant boost in classification performance.
![es-5](https://github.com/user-attachments/assets/41791f9a-ae15-49c7-8e55-56058f7fb7af)


## Conclusion
This project provided an introduction to both Computer Vision and Natural Language Processing. Through hands-on implementation, we explored key machine learning concepts such as linear and logistic regression, artificial neural networks (ANNs), and loss functions. In the CV module, we learned to process facial images using Convolutional Neural Networks (CNNs) and advanced architectures like ResNet, gaining insight into feature extraction and model fine-tuning.

On the NLP side, we explored text vectorization techniques including TF-IDF, and understood how these representations can drive emotion classification tasks. The project also introduced multimodal fusion strategies, specifically late fusion, highlighting how diverse modalities can be combined effectively to enhance predictive performance.

## References
1. Dataset Paper Link: [https://arxiv.org/pdf/1810.02508.pdf](https://arxiv.org/pdf/1810.02508.pdf)
2. [An Assessment of In-the-Wild Datasets for Multimodal Emotion Recognition](https://www.researchgate.net/publication/371195884_An_Assessment_of_In-the-Wild_Datasets_for_Multimodal_Emotion_Recognition)
