import os
import random
import string
import streamlit as st
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np

# Load the saved model
model = load_model("model/predictWaste12.h5")
output_class = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]

def waste_prediction(new_image):
    try: 
        test_image = image.load_img(new_image, target_size = (224,224))
        test_image = image.img_to_array(test_image) / 255
        test_image = np.expand_dims(test_image, axis=0)

        predicted_array = model.predict(test_image)
        predicted_value = output_class[np.argmax(predicted_array)]
        predicted_accuracy = round(np.max(predicted_array) * 100, 2)
        dangerous = ["battery", "plastic", "trash"]
        # Add suggestions for how to treat the waste based on predicted value
        if predicted_value == "battery":
            suggestion = "Batteries should be recycled at a proper recycling facility. Do not throw them in the trash."
        elif predicted_value == "biological":
            suggestion = "Biological waste can be composted to create nutrient-rich soil for plants."
        elif predicted_value == "brown-glass":
            suggestion = "Brown glass can be recycled. Rinse out the container and remove any lids or caps before recycling."
        elif predicted_value == "cardboard":
            suggestion = "Cardboard can be recycled. Flatten the cardboard and remove any tape or labels before recycling."
        elif predicted_value == "clothes":
            suggestion = "Clothes can be donated to a charity or thrift store. If they are too worn out to donate, they can be recycled into rags or insulation."
        elif predicted_value == "green-glass":
            suggestion = "Green glass can be recycled. Rinse out the container and remove any lids or caps before recycling."
        elif predicted_value == "metal":
            suggestion = "Metal can be recycled. Rinse out the container and remove any labels or lids before recycling."
        elif predicted_value == "paper":
            suggestion = "Paper can be recycled. Remove any plastic or metal components before recycling."
        elif predicted_value == "plastic":
            suggestion = "Plastic can be recycled. Rinse out the container and remove any labels or lids before recycling."
        elif predicted_value == "shoes":
            suggestion = "Shoes can be donated to a charity or thrift store. If they are too worn out to donate, they can be recycled into new products."
        elif predicted_value == "trash":
            suggestion = "Trash should be disposed of in a proper waste bin. Try to reduce the amount of trash you produce by recycling and composting."
        elif predicted_value == "white-glass":
            suggestion = "White glass can be recycled. Rinse out the container and remove any lids or caps before recycling."
    except Exception as e:
        st.error("Error loading image")
        st.error(e)
        return
    return f"Your waste material is {predicted_value} with {predicted_accuracy}% accuracy.", suggestion, predicted_value in dangerous


st.set_page_config(page_title="Trash Classification", page_icon="static/logo.jpg")

col1, col2 = st.columns([1, 3])

with col1:
    st.image(Image.open("static/logo.jpg"), width=100)
with col2:
    st.title("Trash Classification")
    st.write("Upload a garbage image and get predictions")
    
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + '.png'
    filepath = os.path.join('images', filename)
    with open(filepath, 'wb') as f:
        f.write(uploaded_file.read())

    image_loaded = cv2.imread(filepath)
    image_loaded = cv2.cvtColor(image_loaded, cv2.COLOR_BGR2RGB)
    st.image(image_loaded, caption="Uploaded Image", use_column_width=True, width=50)

    result = waste_prediction(filepath)
    os.remove(filepath)
    st.success(result[0])
    if result[2] == True:
        st.error("CRITICAL waste material detected!")
        st.warning(result[1])
    else:
        st.info(result[1])
