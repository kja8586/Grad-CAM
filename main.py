from model_config_tf import model_builder, img_shape, preprocess_input, last_conv_layer, img_path
from img_preprocess import get_img_arr
from heatmap import grad_cam_heatmap
from image_impose import display_gradcam

import matplotlib.pyplot as plt

img_arr = preprocess_input(get_img_arr(img_path, img_shape))

model = model_builder(weights="imagenet")

model.layers[-1].activation = None

heatmap = grad_cam_heatmap(img_arr, model, last_conv_layer)

plt.matshow(heatmap)
plt.show()

display_gradcam(img_path, heatmap)