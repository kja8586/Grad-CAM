Grad-CAM

Grad-CAM stands for Gradient-Weighted Class Activation Mapping.
This technique produces visual explanations for decisions made by convolutional neural networks (CNNs).

It answers the question:

“Which regions of the image were most important for this specific class prediction?”

Grad-CAM works by computing the gradients of a target concept (e.g., a classification label) with respect to the feature maps of the final convolutional layer. It then generates a heatmap highlighting important regions in the image.