import tensorflow as tf
import keras

model_builder = Xception  # Assign class directly not the instance
img_shape = (299, 299)

preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

last_conv_layer = "block14_sepconv2_act"

img_path = "/content/drive/MyDrive/img.jpeg" # Replace with available image
