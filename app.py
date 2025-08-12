import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Bangla Sign Language Recognition",
    page_icon="👋",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .prediction-text {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
    }
    .centered-image {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">Bangla Sign Language Recognition for Children</h1>', unsafe_allow_html=True)

# Load the VGG16 model
@st.cache_resource
def load_vgg_model():
    try:
        model = load_model('vgg16_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Process and predict function
def predict_sign(img, model):
    # Ensure image is in RGB mode (3 channels)
    img = img.convert('RGB')
    
    # Resize image to 125x125 as required by the model
    img = img.resize((125, 125))
    
    # Convert to array and preprocess
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    prediction = model.predict(img_array)
    
    return prediction

# Get class names based on the model output
def get_class_name(prediction):
    # Class names provided by user
    class_names = [
        'অনুরোধ', 'আজ', 'আমি', 'খারাপ', 'গরু', 'গৃহ', 'ঘর', 'ঘুম', 'জুতা', 'তুমি', 'ত্বক', 'বন্ধু', 'বাটি', 'ভালো', 'মুরগী', 'সাহায্য'
    ]
    
    # Get the predicted class index
    class_index = np.argmax(prediction)
    
    # Return the class name
    if class_index < len(class_names):
        return class_names[class_index]
    else:
        return "Unknown"

def main():
    # Load the model
    model = load_vgg_model()
    if not model:
        st.warning("Failed to load model. Please check the model file.")
        return
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Upload an Image</h2>', unsafe_allow_html=True)
        
        # Image upload option
        uploaded_file = st.file_uploader("Choose a sign language image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', width=300)
            
            # Button to predict
            if st.button("Recognize Sign"):
                with st.spinner('Processing...'):
                    # Add a small delay to show the spinner
                    time.sleep(1)
                    
                    # Make prediction
                    prediction = predict_sign(img, model)
                    
                    # Get class name
                    class_name = get_class_name(prediction)
                    
                    # Display result
                    st.markdown(f'<div class="prediction-text">Predicted Sign: {class_name}</div>', unsafe_allow_html=True)
                    
                    # Display confidence
                    confidence = float(np.max(prediction) * 100)
                    st.progress(confidence / 100)
                    st.write(f"Confidence: {confidence:.2f}%")
    
    with col2:
        st.markdown('<h2 class="sub-header">Try Sample Images</h2>', unsafe_allow_html=True)
        
        # Sample images section
        sample_dir = 'images'
        if os.path.exists(sample_dir):
            sample_images = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if sample_images:
                # Let user select a sample image
                selected_sample = st.selectbox("Select a sample image:", sample_images)
                
                if selected_sample:
                    # Display the selected sample image
                    sample_img_path = os.path.join(sample_dir, selected_sample)
                    sample_img = Image.open(sample_img_path)
                    st.image(sample_img, caption=f'Sample: {selected_sample}', width=300)
                    
                    # Button to predict from sample
                    if st.button("Recognize Sample"):
                        with st.spinner('Processing...'):
                            # Add a small delay to show the spinner
                            time.sleep(1)
                            
                            # Make prediction
                            prediction = predict_sign(sample_img, model)
                            
                            # Get class name
                            class_name = get_class_name(prediction)
                            
                            # Display result
                            st.markdown(f'<div class="prediction-text">Predicted Sign: {class_name}</div>', unsafe_allow_html=True)
                            
                            # Display confidence
                            confidence = float(np.max(prediction) * 100)
                            st.progress(confidence / 100)
                            st.write(f"Confidence: {confidence:.2f}%")
            else:
                st.info("No sample images found in the images directory.")
        else:
            st.info("Sample images directory not found.")
    
    # Information section at the bottom
    st.markdown('---')
    st.markdown('<h3 class="sub-header">About Bangla Sign Language</h3>', unsafe_allow_html=True)
    st.write("""
        This application helps recognize Bangla Sign Language gestures. 
        Upload your own image or select from our sample images to try it out!
        
        The system is designed to be child-friendly and educational.
    """)

if __name__ == "__main__":
    main()
