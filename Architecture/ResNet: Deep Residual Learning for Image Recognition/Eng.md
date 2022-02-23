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

  - **Vanishing/Exploding gradients**
    - This problem has been largely addressed by normalized initialization & intermediate normalization layers
    
  - When deeper networks can start converging, a ***degradation*** has been exposed:
    - with the network depth increasing, accuracy gets saturated and then degrades rapidly
    
    - Not caused by overfitting, and adding more layers to a suitably deep model leads to higher ***training*** error.
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155293529-9fac8ef1-e65f-47d3-87b7-1ad5c077ee76.png" width = "40%" height = "40%"></p>


- It shows that all systems are hard to optimize.

- Consider the shallow architecture and its deeper counterpart that adds more layers onto it. 

- Suppose the learned shallow architecture is optimal. Then, there exists a solution for deeper model that the added layers are identity mapping.

- So, the existence of this solution indicates that a deepermodel should produce no higher training error than its shallower counterpart, but it doesn't.

- Introduce a ***deep residual learning*** framework
  - Denoting the desired underlying mapping as *H(x)*
  
  - Let the stacked nonlinear layers fit another mapping of ***F(x) = H(x) - x***
  
  - Hypothesis: it is easier to optimize the residual mapping than to optimize the original.
  
  - To the extreme, if an identity mapping(*H(x) = x*) were optimal, it would be easier to push the residual to zero.
	<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155295564-a2affa73-ae8e-4b0f-981a-741bd4b246c9.png" width = "40%" height = "40%"></p>

  
	- Short connection
		- simply perform *identity* mapping and their outputs added to the output of the stacked layers.
		- it requires neither extra parameter nor computational complexity
- We show that
	- 1) Extremely ResNets are easy to optimize, but the counterpart "plain" nets exhibit higher training error when the depth increases
	
	- 2) ResNets can easily enjoy accuracy gains from greatly increased depth, producing results substantially better than previous networks.


## 3. Deep Residual Learning
#### 3.1. Residual Learning
- If one hypothesizes that multiple nonlinear layers can asymptotically approximate complicated fnctions, then it is equivalent to hypothesize that they can asymptotically approximate the residual functions, *F(x) = H(x) - x*. 

- They are only mathematically transposing, but the ease of learning varies depending on the form.

- This reformulation is motivated by the counterintuitive phenomena about the degradation problem in the introduction.
	- The degradation problem suggests that the solvers might have difficulties in approximating identity mappings with multiple nonlinear layers.
	- With the residual learning reformulation, if identity mappings are optimal, the solvers may simply drive the weights of the multiple nonlinear layers toward zero to approach identity mappings.

#### 3.2. Identity Mapping by Shortcuts
- Adopt residual learning to every few stacked layers. 

- We consider a building block defined as <img src = "https://user-images.githubusercontent.com/88715406/155332522-70200c27-341c-4c0e-9696-99766868cf84.png" width = "10%" height = "10%">
	- *F* represents the residual mapping to be learned. 
	- ex) In Figure 2, <img src = "https://user-images.githubusercontent.com/88715406/155332741-29aff029-f461-45c5-8b1d-8819548d5615.png" width = "10%" height = "10%"> where sigma is ReLU and the biases are omitted for simplifying notations.


- It introduces neither extra parameter nor computation complexity.
	- can fairly compare plain/residual networks that simultaneously have the same number of parameters, depth, width, and computational cost (except for the negligible element-wise addition).
	
- Also the dimensions of ***x*** and ***F*** must be equal. 
	- If not, perform a linear projection ***W_s*** to match the dimensions.
	
- The form of the residual function ***F*** is flexible.
	- it can have 2, 3 or more layers.
	
	- But if it has only a single layer, it is similar to linear layer ***Wx + x***.
	
	- Also applicable to convolutional layers.

#### 3.3. Network Architectures
###### Plain Network
- Inspired by VGG nets

- The convolutional layers mostly have 3X3 filters and follow 2 simple design rules:
	- 1) for the same output feature map size, the layers have the same number of filters.
	
	- 2) if the feature map size is halved, the number of filters is doubled to preserve the time complexity per layer.
	
- Perform downsampling directly by convolutional layers that have a stride of 2.
- Ends with a global average pooling layer and a 1000-way fc layer with softmax.

<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155338853-0f3c484d-46e8-4128-8939-ec10f8b31f3e.png" width = "40%" height = "40%"></p>

#### 3.4. Implementation
- Scale Augmentation : random resize with its shorter side in 256 ~ 480
- Color Augmentation : used in AlexNet
- Batch Normalization : right after each Conv Layer & before activation
- Weights Initialization
- SGD : with a mini-Batch 256
- Learning Rate : starts from 0.1 and is divided by 10 when the error plateaus
- Iteration : 60 x 10^4
- Decay & Momentum : 0.0001, 0.9
- No Dropout 
