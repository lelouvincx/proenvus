from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


# Create a class to hold the model and its methods
class Model:
    def __init__(self, model_path, output_class):
        self.model = load_model(model_path)
        self.output_class = output_class

    def waste_prediction(self, new_image):
        try: 
            test_image = image.load_img(new_image, target_size = (224,224))
            test_image = image.img_to_array(test_image) / 255
            test_image = np.expand_dims(test_image, axis=0)

            predicted_array = self.model.predict(test_image)
            predicted_value = self.output_class[np.argmax(predicted_array)]
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
