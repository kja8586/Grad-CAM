\section{Grad CAM}
Grad CAM stands for Gradient-Weighted Class Activation Mapping. This technique is used to produce visual explanations for decisions made by convolutional neural networks (CNNs). By calculating gradients of a target concept (e.g., a classification label) with respect to the final convolutional layer's feature maps, it generates a heat map highlighting important regions in an image. 

\textbf{“Which regions of the image were most important for this specific class prediction?”}

\subsection{Terminology}
\begin{itemize}
    \item Let $y^c$ denote the score (logit) for class $c$.
    \item Let $A^k$ denote the $k$-th feature map of the chosen convolutional layer.
    \item Let $A_{ij}^k$ denote the activation at spatial location $(i, j)$ in the $k$-th feature map.

\end{itemize}

