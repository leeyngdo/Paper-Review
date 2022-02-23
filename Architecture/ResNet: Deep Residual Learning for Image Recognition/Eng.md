# Deep Residual Learning for Image Recognition
----------------------------------------------------------------------------------------------------------------------------------------------------
**Table of Contents**

* [**Abstract**](#abstract)

* **1.** [**Introduction**](#1-introduction)

* **3.** [**Deep Residual Learning**](#3-deep-residual-learning)
	- **3.1.** [**Residual Learning**](#31-residual-learning)
	- **3.2.** [**Identity Mapping by Shortcuts**](#32-identity-mapping-by-shortcuts)
  - **3.3.** [**Network Architecture**](#33-network-architectures)
  - **3.4.** [**Implementation**](#34-implementation)
* **4.** [**Experiments**](#4-experiments)
	- **4.1.** [**ImageNet Classification**](#41-unsupervised-contrastive-learning-benefits-more-from-bigger-models)
	- **4.2.** [**CIFAR-10 and Analysis**](#42-cifar-10-and-analysis)

----------------------------------------------------------------------------------------------------------------------------------------------------
## Abstract
- Deeper neural network are more difficult to train.
- Present ResNet to make it easy.
  - Reformulate the layers as learning residual functions with reference to the layer inputs.
  - These are easier to optimize & can gain accuracy from considerably increased depth.

## 1. Introduction
- Deep networks naturally integrate low/mid/high level features [M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014.] and classifiers in an end-to-end multi-layer fashion, the “levels” of features can be enriched
by the number of stacked layers (depth).
  - It is known that the deeper model performs better because of many of the papers referenced by this paper. (And there are many studies on CNN.
  
- But, ***Is learning better networks as easy as stacking more layers?***
  - Vanishing/Exploding gradients
    - This problem has been largely addressed by normalized initialization & intermediate normalization layers
  - When deeper networks can start converging, a ***degradation*** has been exposed:
    - with the network depth increasing, accuracy gets saturated and then degrades rapidly
    - Not caused by overfitting, and adding more layers to a suitably deep model leads to higher ***training*** error.
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155293529-9fac8ef1-e65f-47d3-87b7-1ad5c077ee76.png" width = "40%" height = "40%"></p>


- It shows that all systems are hard to optimize.

- Introduce a ***deep residual learning*** framework
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155295564-a2affa73-ae8e-4b0f-981a-741bd4b246c9.png" width = "40%" height = "40%"></p>
  
  
  - Denoting the desired underlying mapping as *H(x)*
  - Let the stacked nonlinear layers fit another mapping of ***F(x) = H(x) - x***
  - Hypothesis: it is easier to optimize the residual mapping than to optimize the original.
  - To the extreme, if an identity mapping(*H(x) = x*) were optimal, it would be easier to push the residual to zero.

	
