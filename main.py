import os, random, string
import streamlit as st
import cv2
from PIL import Image

from model import Model


def show_input_prompt():
    st.set_page_config(page_title="Trash Classification", page_icon="static/logo.jpg")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(Image.open("static/logo.jpg"), width=100)
    with col2:
        st.title("Trash Classification")
        st.write("Upload a garbage image and get predictions")

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])

    return uploaded_file


if __name__ == "__main__":
    # Create ML model
    model_path = "model/predictWaste12.h5"
    output_class = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]
    model = Model(model_path, output_class)

    uploaded_file = show_input_prompt()

    if uploaded_file is not None:
        # Create a random filename
        filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png'
        filepath = os.path.join('images', filename)
        # Save the uploaded file to the filename
        with open(filepath, 'wb') as f:
            f.write(uploaded_file.read())

        # Read from the filepath and display the image
        image_loaded = cv2.imread(filepath)
        image_loaded = cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB)
        st.image(image_loaded, caption="Uploaded Image", use_column_width=True, width=50)

        result = model.waste_prediction(filepath)
        st.write(result)
        st.success(result[0])
        if result[2] == True:
            st.error("CRITICAL waste material detected!")
            st.warning(result[1])
        else:
            st.info(result[1])

        os.remove(filepath)