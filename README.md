# Bangla Sign Language Recognition (Children)

 This project is a Streamlit web application that recognizes Bangla Sign Language gestures from images. It is designed to be simple and child‑friendly.

 ## Overview
 - Uses a pre‑trained Keras model saved at `vgg16_model.h5`.
 - Allows two ways to test:
   - Upload any image (JPG/PNG)
   - Pick a sample image from `images/` directory
 - Shows predicted Bangla label and confidence.

 ## Model
 - File: `vgg16_model.h5`
 - Expected input: `125 x 125 x 3` (RGB)
 - Output: probabilities over 16 classes.

 ## Bangla Class Labels (index order)
 1. অনুরোধ  
 2. আজ  
 3. আমি  
 4. খারাপ  
 5. গরু  
 6. গৃহ  
 7. ঘর  
 8. ঘুম  
 9. জুতা  
 10. তুমি  
 11. ত্বক  
 12. বন্ধু  
 13. বাটি  
 14. ভালো  
 15. মুরগী  
 16. সাহায্য

 Make sure this order matches the model’s training order. If predictions look shifted, adjust the list in `get_class_name()` accordingly.

 ## Image Preprocessing (important)
 - Always convert to RGB: `img.convert('RGB')`
 - Resize to `125 x 125`
 - Convert to array and normalize: `img_array / 255.0`

 ## How to Run
 ```bash
 pip install -r requirements.txt
 streamlit run app.py
 ```
 Then open http://localhost:8501

 ## Directory Structure
 ```
 d:\\Coder\\Sign\\
 ├─ app.py                 # Streamlit app
 ├─ vgg16_model.h5         # Trained model
 ├─ images\\                # Sample images
 ├─ requirements.txt       # Dependencies
 └─ README.md / note.md    # Docs
 ```

 ## Troubleshooting
 - Input shape error (expected 125x125x3): ensure resize to 125x125.
 - Channel depth error (1 vs 3): ensure `convert('RGB')` before resizing.
 - Progress bar type error: cast to Python float before passing to `st.progress`.
 - Model not found: place `vgg16_model.h5` next to `app.py` or update path.
 - Wrong labels: verify class order matches training.

 ## Tech Stack
 - Streamlit, TensorFlow/Keras, NumPy, Pillow

 ## Next Steps / Ideas
 - Add camera capture support
 - Add Bengali/English UI toggle and voice feedback
 - Improve layout and colors for kids
 - Batch prediction for multiple images
 - Confusion matrix view for dataset evaluation

 ## Credits
 Dataset images and model provided by the project owner. App scaffolding created to run locally on Windows.