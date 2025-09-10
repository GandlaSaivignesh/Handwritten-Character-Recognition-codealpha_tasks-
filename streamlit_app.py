import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from predict import predict_digit
import tempfile
import os

st.title("Handwritten Character Recognition")

uploaded_file = st.file_uploader("Upload an image of a handwritten digit", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict the digit
    if st.button("Predict"):
        predicted_digit = predict_digit(tmp_path)
        st.write(f"Predicted Digit: {predicted_digit}")

    # Clean up the temporary file
    os.unlink(tmp_path)
