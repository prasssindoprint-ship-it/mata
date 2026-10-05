import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 1. Page Config & Title
st.set_page_config(page_title="ResNet Image Classifier", layout="centered")
st.title("🖼️ ResNet50 Keras Classifier")
st.write("Upload an image to see what the ResNet50 model predicts!")

# 2. Cache the model load so it runs fast after the first startup
@st.cache_resource
def load_resnet_model():
    # To use a custom local model instead, use: tf.keras.models.load_model('my_model.h5')
    return ResNet50(weights='imagenet')

with st.spinner("Loading Keras Model..."):
    model = load_resnet_model()

# 3. File Uploader UI Widget
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # 4. Image Preprocessing
    # ResNet50 expects 224x224 images
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # 5. Inference
    with st.spinner("Classifying..."):
        predictions = model.predict(img_array)
        decoded = decode_predictions(predictions, top=3)[0]
    
    # 6. Display Results
    st.subheader("Top Predictions:")
    for i, (imagenet_id, label, prob) in enumerate(decoded):
        st.write(f"**{i+1}. {label.replace('_', ' ').title()}**: {prob*100:.2f}%")
