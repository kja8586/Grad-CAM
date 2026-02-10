def get_img_arr(img_path, size):
  img = keras.utils.load_img(img_path, target_size=size)  # Load the image and reshape it. img is PIL object
  img = keras.utils.img_to_array(img) # Convert PIL object to array
  img = np.expand_dims(img, axis=0) # Adding batch dimesion so that it can be passed to the model

  return img
