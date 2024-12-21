import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load the trained model (ensure the path is correct)
model = tf.keras.models.load_model('mnist_cnn_model.keras')

# File uploader in Streamlit to allow users to upload an image
st.title("Handwritten Digit Recognition")
st.markdown("upload an image of a handwritten digit, and the model will predict which digit it is!(0-9)")
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    # Display the original uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    # Step 1: Convert to grayscale
    image = image.convert('L')  # Convert to grayscale (MNIST format is grayscale)

    # Step 2: Resize to 28x28 pixels
    image = image.resize((28, 28))

    # Step 3: Invert colors (MNIST is white digits on black background)
    # Invert only if the background is not black
    if np.mean(image) > 127:
        image = ImageOps.invert(image)

    # Step 4: Convert image to a numpy array
    image = np.array(image, dtype=np.float32)

    # Step 5: Normalize pixel values to the range [0, 1]
    image = image / 255.0

    # Step 6: Reshape image to match model input shape
    image = image.reshape(1, 28, 28, 1)

    # Display preprocessed image to verify if it’s correct
    st.image(image.reshape(28, 28), caption="Preprocessed Image", use_column_width=True, clamp=True)

    # Predict the digit
    prediction = model.predict(image)
    
    # Display the full prediction array for debugging
    st.write(f"Prediction Array: {prediction}")
    
    # Get the digit with the highest probability
    predicted_digit = np.argmax(prediction)
    
    # Display the result
    st.write(f"Predicted Digit: {predicted_digit}")
