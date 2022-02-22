# A Simple Framework for Contrastive Learning of Visual Representations

## Abstract
- This paper presents SimCLR: a simple framework for contrastive learning of visual representations.
  - Simplified proposed contrastive selfsupervised learning algorithms without requiring specialized architectures or a memory bank.
- What enables the contrastive prediction tasks to learn useful representations?
	- (1) Composition of data augmentations
	- (2) Introduce a learnable nonlinear transformation 
	- (3) Larger batch sizes and mor training steps
- By combining this, we can considerably outperform previous method for self-supervised & semi-supervied learning on ImageNet.
<p align="center"><img src = "https://user-images.githubusercontent.com/88715406/155086702-17a7af0f-5e85-4098-8caf-370860305411.png" width = "60%" height = "60%"></p>

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
	- (1) Composition of multiple data augmentation operations
	- (2) Learnable nonlinear transformation between the representation and the contrastive loss
	- (3) Representation learning with contrastive cross entropy
loss
	- (4) Larger batch sizes and Longer training

