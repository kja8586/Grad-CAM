# Grad-CAM

Grad-CAM stands for Gradient-Weighted Class Activation Mapping.
This technique produces visual explanations for decisions made by convolutional neural networks (CNNs).

It answers the question:
```
“Which regions of the image were most important for this specific class prediction?”
```
Grad-CAM works by computing the gradients of a target concept (e.g., a classification label) with respect to the feature maps of the final convolutional layer. It then generates a heatmap highlighting important regions in the image.

** Terminology **
Let 𝑦^𝑐 denote the score (logit) for class c.
Let 𝐴^𝑘 denote the k-th feature map of the chosen convolutional layer.
Let A_{ij}^k denote the activation at spatial location (i, j) in the k-th feature map.
Let 𝑍 denote the total number of spatial locations in a feature map.