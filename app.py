import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import openai  # Import OpenAI for GenAI
import os
from PIL import Image
import base64
import io

# Load OpenAI API key (Set your API key here)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://drive.google.com/file/d/1eyOOzo1wPdVhBubrFlcpuz8RUOUomdFk/view?usp=drive_link"  # Replace with your file ID
    model_path = "model.h5"

    if not os.path.exists(model_path):
        gdown.download(model_url, model_path, quiet=False)

    model = tf.keras.models.load_model(model_path)

    # Define input shape and manually build the model
    dummy_input = np.zeros((1, 120, 120, 3), dtype=np.float32)
    model.predict(dummy_input)  # Ensures the model is init ialized

    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure it's RGB
    image = image.resize((120, 120))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension
    return image

# Function to create a downloadable report
def create_download_link(image, prediction, probability):
    result = "Pneumonia Detected" if probability > 0.5 else "Normal X-ray"
    text = f"Prediction: {result}\nConfidence: {probability * 100:.2f}%"

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()

    # Encode image to Base64
    img_base64 = base64.b64encode(img_byte_arr).decode()
    
    # HTML link for downloading
    href = f'<a href="data:image/png;base64,{img_base64}" download="prediction.png">Download Prediction Image</a><br><br>'
    href += f'<a href="data:text/plain;charset=utf-8,{text}" download="prediction.txt">Download Report</a>'
    return href

import openai

def get_ai_response(user_input):
    try:
        client = openai.OpenAI()  # New OpenAI client object

        response = client.chat.completions.create(
            model="gpt-4",  # You can use "gpt-3.5-turbo" if needed
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant providing information about pneumonia."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        st.sidebar.error("⚠️ OpenAI API error. Check your API key or connection.")
        st.sidebar.write(f"Debug Info: {str(e)}")
        return "Sorry, I couldn't process your request. Please try again later."

# Streamlit UI
st.title("Pneumonia Detection with AI Chatbot")
st.write("Upload a Chest X-ray Image to classify it as **Normal or Pneumonia**.")

# File Uploader for User's Image
uploaded_file = st.file_uploader("Choose an X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict
    prediction = model.predict(preprocessed_image)
    probability = prediction[0][0]

    # Display results
    st.subheader("Prediction:")
    st.subheader("Prediction:")
    if probability > 0.5:
        result = "Pneumonia Detected"
        confidence = probability * 100
        st.error(f"⚠️ {result} with {confidence:.2f}% confidence")
    else:
        result = "Normal Chest X-ray"
        confidence = (1 - probability) * 100
        st.success(f"✅ {result} with {confidence:.2f}% confidence")

    # ✅ Use the same confidence in the download link
    st.markdown(create_download_link(image, result, confidence/100), unsafe_allow_html=True)

# **Chatbot UI (AI-powered)**
st.sidebar.title("💬 Pneumonia AI Chatbot")
st.sidebar.write("Ask me anything about pneumonia!")

user_question = st.sidebar.text_input("Type your question here...")

if user_question:
    response = get_ai_response(user_question)
    st.sidebar.write(f"🤖 **Chatbot:** {response}")
