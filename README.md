# Grad-CAM

**Grad-CAM (Gradient-Weighted Class Activation Mapping)** is a technique used to produce visual explanations for decisions made by Convolutional Neural Networks (CNNs).

It answers the question:

> **Which regions of the image were most important for this specific class prediction?**

Grad-CAM works by computing gradients of a target class score with respect to the feature maps of the final convolutional layer and producing a heatmap highlighting important regions.

---

## Terminology

-  $y^c$  — score (logit) for class ( $c$ )
-  $A^k$  — ( $k$ )-th feature map of the selected convolutional layer
-  $A_{ij}^k$  — activation at spatial location ( $(i, j)$ ) in feature map ( $k$ )
-  $Z$  — total number of spatial locations in a feature map


---

## Algorithm Steps

### 1️⃣ Compute Gradients

Compute the gradient of the class score with respect to each feature map activation:

```math
\frac{\partial y^c}{\partial A_{ij}^k}
```

This tells us how sensitive the class score is to each spatial location.

---

### 2️⃣ Global Average Pooling (Importance Weights)

Compute importance weights for each feature map:

```math
\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}
```

- `α_k^c` represents how important feature map `k` is for class `c`.

---

### 3️⃣ Weighted Combination

Combine feature maps using the computed weights:

```math
L_{Grad-CAM}^c = \sum_k \alpha_k^c A^k
```

---

### 4️⃣ Apply ReLU

```math
Grad\text{-}CAM^c = ReLU(L_{Grad-CAM}^c)
```

ReLU keeps only positive contributions (regions that support the class prediction).

---

## Final Output

- A coarse localization heatmap
- Upsampled to input image size
- Overlaid on the original image for visualization

---

## Summary

Grad-CAM:
- Uses gradients flowing into the last convolutional layer
- Produces class-discriminative localization maps
- Works with most CNN-based architectures
- Does not require architectural modifications

---

## Quick Start
**Cloning of repository**
```
git clone https://github.com/kja8586/Grad-CAM.git
```
**Changing directory**
```
cd Grad-CAM
```
**Installing dependencies**
Using requirements.txt
``` 
pip install -r requirements.txt
```

**Execute the code**
```
python main.py
```