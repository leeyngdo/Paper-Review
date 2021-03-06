# A Simple Framework for Contrastive Learning of Visual Representations
----------------------------------------------------------------------------------------------------------------------------------------------------
**Table of Contents**

* [**Abstract**](#abstract)

* **1.** [**Introduction**](#1-introduction)

* **2.** [**Method**](#2-method)
	- **2.1.** [**The Contrastive Learning Framework**](#21-the-contrastive-learning-framework)
	- **2.2.** [**Training with Large Batch Size**](#22-Training-with-Large-Batch-Size)

* **3.** [**Data Augmentation for Contrastive Representation Learning**](#3-data-augmentation-for-contrastive-representation-learning)
	- **3.1.** [**Composition of data augmentation operations is crucial for learning good representations**](#31-composition-of-data-augmentation-operations-is-crucial-for-learning-good-representations)
	- **3.2.** [**Contrastive learning needs stronger data augmentation than supervised learning**](#32-contrastive-learning-needs-stronger-data-augmentation-than-supervised-learning)

* **4.** [**Architectures for Encoder and Head**](#4-architectures-for-encoder-and-head)
	- **4.1.** [**Unsupervised contrastive learning benefits (more)
from bigger models**](#41-unsupervised-contrastive-learning-benefits-more-from-bigger-models)
	- **4.2.** [**A nonlinear projection head improves the
representation quality of the layer before it**](#42-a-nonlinear-projection-head-improves-the-representation-quality-of-the-layer-before-it)

* **5.** [**Loss Functions and Batch Size**](#5-loss-functions-and-batch-size)
	- **5.1.** [**Normalized cross entropy loss with adjustable temperature works better than alternatives**](#51-normalized-cross-entropy-loss-with-adjustable-temperature-works-better-than-alternatives)
	- **5.2.** [**Contrastive learning benefits (more) from larger batch sizes and longer training**](#52-contrastive-learning-benefits-more-from-larger-batch-sizes-and-longer-training)

* **6.** [**Comparison with State-of-the-art**](#6-comparison-with-state-of-the-art)
* **8.** [**Conclusion**](#8-conclusion)
----------------------------------------------------------------------------------------------------------------------------------------------------
## Abstract
- This paper presents **SimCLR: a simple framework for contrastive learning of visual representations.**
  - Simplified proposed contrastive selfsupervised learning algorithms without requiring specialized architectures or a memory bank.
- What enables the contrastive prediction tasks to learn useful representations?
	- **(1)** Composition of data augmentations
	- **(2)** Introduce a learnable nonlinear transformation 
	- **(3)** Larger batch sizes and more training steps
- By combining this, we can considerably outperform previous method for self-supervised & semi-supervied learning on ImageNet.
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155086702-17a7af0f-5e85-4098-8caf-370860305411.png" width = "40%" height = "40%"></p>

## 1. Introduction
#### Learning effective visual representations without human supervision is a long-standing problem.

- Most main approaches belong to one of two classes: generative / discriminative
	- generative
		- learn to generate model pixels in the input space
		- But, pixel-level generation is expensive
		- Not necessary for representation learning. 
	- discriminative
		- learn representations using objective functions 
		- but but train networks to perform pretext tasks where both the inputs and labels are derived from an unlabeled dataset.  
- We introduce a SimCLR
	- Outperform prev work
	- Simpler
	- Not requiring neither specialized architectures nor a memory bank
#### What enables good contrastive representation learning?
- We systematically study the major components of our framework and show that:
	- **(1)** Composition of multiple data augmentation operations
	- **(2)** Learnable nonlinear transformation between the representation and the contrastive loss
	- **(3)** Representation learning with contrastive cross entropy
loss
	- **(4)** Larger batch sizes and Longer training
	
## 2. Method
#### 2.1. The Contrastive Learning Framework
- SimCLR learns representations by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space.
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155092881-81285538-e766-40db-9f96-d7dc8ffb9312.png" width = "40%" height = "40%"></p>

- **(1)** stochastic data augmentation module
	- Transforms any given data randomly resulting in 2 correlated views of the same example, denoted ~x_i and ~x_j (**positive pair**)
	- Applied three augmentations: ***random cropping followed by resize back to the original size, random color distortions, and random Gaussian blur***
	
- **(2)** A neural network ***base encoder f()*** that extracts representation vectors from augmented data examples.
	- Our framework allows any choices of the network architecture
	- We chose for simplicity and adopt the commonly used ResNet to obtain <img src = "https://user-images.githubusercontent.com/88715406/155159727-ee52c9e6-203b-437d-ba06-fd3a7536d679.png" width = "20%" height = "20%"> 
	
		where h is the output after the average pooling layer.
		
- **(3)** A small neural network ***projection head g()*** that maps representations to the space where contrastive loss is
applied.
	- Used a MLP with one hidden layer to obtain <img src = "https://user-images.githubusercontent.com/88715406/155161674-1efd9eb6-2332-45db-8c1b-281ba4ce3368.png" width = "20%" height = "20%"> 
		where sigma is a ReLU non-linearity
		
- **(4)** A ***contrastive loss function*** defined for a contrastive prediction task.
	- Given a set {x??k} including a positive pair of examples x??i and x??j , the ***contrastive prediction task*** aims to identify x??j in {x??k}k!=i for a given x??i.

- We randomly sample a minibatch of N examples and  define
the contrastive prediction task on pairs of augmented examples derived from the minibatch, resulting in 2N data points. 
	- We don't sample negative examples explicitly.
	- We treat the other 2(N ??? 1) augmented examples within a minibatch as negative examples.
	- Define the **similarity function** <img src = "https://user-images.githubusercontent.com/88715406/155163850-f7b528e9-1b3d-4aaa-bd64-0711fb71b50f.png" width = "20%" height = "20%">
	- **Loss function(NT-Xent)** <img src = "https://user-images.githubusercontent.com/88715406/155164228-4cfafb2e-05e7-45c0-a170-4b247ae26c7d.png" width = "20%" height = "20%">
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155164718-88ab1847-0ae4-4871-b7e3-96627ec25a09.png" width = "40%" height = "40%"></p>

#### 2.2. Training with Large Batch Size
- We vary the training batch size N from 256 to 8192 
	- A batch size of 8192 gives us 16382 negative examples per positive pair from both augmentation views.

- Standard ResNets use batch normalization.
	- In distributed training with data parallelism, the BN mean and variance are typically aggregated locally per device. 
	- In our contrastive learning, as positive pairs are computed in the same device, the model can exploit the local information leakage to improve prediction accuracy without improving representations. 




## 3. Data Augmentation for Contrastive Representation Learning
#### Data augmentation defines predictive tasks.
- It has not been considered as a systematic way to define the contrastive prediction task before.
	- Mostly used changing the architecture 
		- Ex) achieve global-to-local view prediction via constraining the receptive field in the network architecture
- But this complexity can be avoided by simple random cropping with resizing of target images. 
- Broader contrastive prediction tasks can be defined by extending the family of augmentations and composing them stochastically.

<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155172500-17769f96-8e92-4091-a963-bebda48bd2e0.png" width = "60%" height = "60%"></p>

#### 3.1. Composition of data augmentation operations is crucial for learning good representations
- Data augmentation we used
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155174019-2da19edd-e564-431a-881a-35df228ed777.png" width = "50%" height = "50%"></p>

- Investigate the performance of our framework when applying augmentation compositions.

- Since ImageNet images are of different sizes(resolutions), we always apply crop and resize images 
	- But makes it difficult to study other augmentations in the absence of cropping 
	- **Solution: asymmetric data transformation setting**
		- always first randomly crop images and resize them to the same resolution, and we then apply the targeted transformation only to one branch of the framework while leaving the other branch
as the identity (i.e. t(x) = x).
		- It may hurts the performance. But this is suboptimal than applying augmentations to both branches, but sufficient for ablation.

- **Result: *no single transformation suffices to learn good representations***
	- When composing augmentations, the contrastive prediction task becomes harder, but the quality of representation improves dramatically.
	
- **Best was random cropping & random color distortion**
	- Without color distortion, one serious issue occurs:
		- most patches from an image share a similar color distribution. <p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155180542-fb5fe435-6fce-4e6c-9d60-b1eb4d6340f0.png" width = "50%" height = "50%"></p>
		- Color histograms alone suffices to distinguish images.
	- It is critical to compose cropping with color distortion in order to learn generalizable features and avoid exploition

#### 3.2. Contrastive learning needs stronger data augmentation than supervised learning
- To further demonstrate the importance of the color augmentation, we adjust the strength.
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155181864-db769f51-7ba8-47b3-9c71-6e4a74ef27c5.png" width = "50%" height = "50%"></p>


- Stronger color augmentation improves the linear evaluation of the learned unsupervised models.
	- AutoAugment(a sophisticated augmentation policy found using supervised learning) doesn't work better for unsupervised.
	- Also stronger color augmentation does not improve or even hurts their performance.
- Thus, unsupervised contrastive learning benefits from stronger (color) data augmentation than supervised learning. 

## 4. Architectures for Encoder and Head
#### 4.1. Unsupervised contrastive learning benefits (more) from bigger models
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155204389-d2564806-1140-4c8a-adbd-909b922db050.png" width = "50%" height = "50%"></p>


- Increasing depth and width both improve performance.
	- Similar findings hold for supervised learning(He et al., 2016)
	- But there is the gap between supervised models and linear classifiers trained on unsupervised models shrinks as the model size
increases, suggesting that unsupervised learning benefits more from bigger models than its supervised counterpart.

#### 4.2. A nonlinear projection head improves the representation quality of the layer before it

- Study the importance of a projection head *g(h)* <p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155269123-da93eaa6-f4f7-41d3-9f04-bcdac761b4a8.png" width = "50%" height = "50%"></p>

- Used 
	- Identity(No projection)
	- Linear projection (as used by several previous approaches)
	- Default nonlinear projection with one additional hidden layer (and ReLU activation)

**- Results: Nonlinear > Linear > Identity** 
	- similar results observed regardless of output dimension 
- Also, the layer before the nonlinear projection head ***h*** is still much better than the layer after ***z = g(h)***.
	- **The hidden layer before the projection head is a better representation than the layer after.**

- So, use the representation before the nonlinear projection. 
	- due to loss of info induced by contrastive loss. 
	- ***z = g(h)*** is trained to be invariant to data transformation.
	- So it can remove information that may be useful for the downstream task e.g. color, orientation of objects

<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155270511-afa1f9a6-48f4-4f90-9567-3dd5364ab569.png" width = "50%" height = "50%"></p>

- Experiments that use either ***h*** or ***g(h)*** to learn
to predict the transformation applied during the pretraining.
	- Set <img src = "https://user-images.githubusercontent.com/88715406/155161674-1efd9eb6-2332-45db-8c1b-281ba4ce3368.png" width = "20%" height = "20%">, with the same input and output dimenstionality.
**- h contains much more info about the transformation applied, while g(h) loses info!**

## 5. Loss Functions and Batch Size
#### 5.1. Normalized cross entropy loss with adjustable temperature works better than alternatives
- Compared the NT-Xent loss against other commonly used contrastive loss. 
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155272002-fa21d12c-6d07-4557-bfc6-3ef5d097a114.png" width = "50%" height = "50%"></p>

- 1) l2 normalization (i.e. cosine similarity) along with temperature effectively weights different examples.
- 2) an appropriate temperature can help the model learn from hard negatives.
	- [More explanation](https://github.com/leeyngdo/Paper-Review/blob/main/Self-Supervised%20Learning/SimCLR:%20A%20Simple%20Framework%20for%20Contrastive%20Learning%20of%20Visual%20Representations/Importance%20of%20Temperature.md)
- 3) unlike cross-entropy, other functions do not weigh the negatives 
#### 5.2. Contrastive learning benefits (more) from larger batch sizes and longer training
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155278807-2f91ee8f-a7b1-48bb-b979-eb5233916c9b.png" width = "50%" height = "50%"></p>

- The impact of batch size when models are trained for different numbers of epochs.
- With small training epochs, larger batch sizes have a significant advantage over the smaller ones.
- With more training steps/epochs, the gaps between different batch sizes decrease or disappear when the batches are randomly resampled. 
- In contrast to supervised learning, larger batch sizes provide more negative examples, facilitating convergence.
	- Training longer provides more negative examples, improving the results. 

## 6. Comparison with State-of-the-art
#### Linear Evaluation
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155278807-2f91ee8f-a7b1-48bb-b979-eb5233916c9b.png" width = "50%" height = "50%"></p>

- Compare with previous approach in the linear evalution setting.

#### Semi-Supervised learning
- Sample 1% or 10% of the labeled ILSVRC-12 training datasets in a class-balanced way (???12.8 and ???128 images per class respectively). 
- Fine-tune the whole base network on the labeled data without regularization. 
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155283918-c44c8bd2-3c02-4640-80db-daff630e8fa2.png" width = "50%" height = "50%"></p>

- Comparisons of results againt recent methods.

#### Transfer Learning
- Evaluate transfer learning performance across 12 natural image datasets in both linear evaluation (fixed feature extractor) and fine-tuning settings.
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155284225-a03bd989-303b-4c82-b536-1c2dca31ff94.png" width = "50%" height = "50%"></p>


## 8. Conclusion
- Present a simple framework and its instantiation for contrastive visual representation learning.
- Carefully study its components, and show the effects of different design choices. 
- By combining findings, we improve considerably over previous methods for selfsupervised, semi-supervised, and transfer learning.
- This approach differs from standard supervised learning on ImageNet only in the choice of data augmentation, the use of a nonlinear head at the end of the network, and the loss function.
