def grad_cam_heatmap(img, model, last_conv_layer, pred_idx=None):
  grad_model = keras.models.Model(
      model.input, [model.get_layer(last_conv_layer).output, model.output]
  ) # Defining model with 2 heads. One return last conv layer output and another give logits or final layer output without softmax

  with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img) # Returns last conv layer filter maps and class score
    if pred_idx is None:
      pred_idx = tf.argmax(preds[0]) # If no class given
    class_channel = preds[:, pred_idx] # Class score from output tensor

    grads = tape.gradient(class_channel, last_conv_layer_output) # Gradient of class wrt feature maps

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)) # Pool grads

    last_conv_layer_output = last_conv_layer_output[0] # Remove batch dimension
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis] # Multiplication of feature maps with pooled gradients
    heatmap = tf.squeeze(heatmap) # Remove batch dimension

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) # Normalize

    return heatmap.numpy()
