---
title: 'Selective Supervised Contrastive Learning with Noisy Labels'
date: 2022-05-10
permalink: /posts/2022/05/selective-supervised-contrastive-learning/
tags:
  - Contrastive Learning
  - Supervised Learning
  - Noisy Labels
---

Contrastive Learning is able to learn good latent representations that can be used to achieve high performance in downstream tasks. Supervised contrastive learning enhances the learned representations using supervised information. However, noisy supervised information corrupts the learned representations. In this blog post, I will summarize and simplify a paper published in CVPR 2022 that proposed an algorithm to learn high quality representations in existence of noisy supervised information. The title of this paper is "Selective Supervised Contrastive Learning with Noisy Labels". 


What is contrastive learning?
------



What is supervised contrastive learning?
------



How to handle noisy labels?
------
The proposed algorithm by the paper "Selective Supervised Contrastive Learning with Noisy Labels" consists of two steps:

* Select pairs of examples that it is confident about their label
* Train the network with the confident pairs using supervised contrastive learning


**How to find confident examples?**

Uses unsupervised training to initially train the network in the first few epochs

Confident examples are found by first measuring cosine distance between the low dimensional representations $z_i$, $z_j$ of each pair of examples

$$d(z_i, z_j) = \frac{z_i z_j^T}{\|\|z_i\|\|~\|\|z_j\|\|}$$

Creating a pseudo-label $\hat{y}_i$ for each example ($x_i$, $\tilde{y}_i$) by aggregating the original label from its top-k neighbors with lowest the cosine distance

$$\hat{q}_c(x_i) = \frac{1}{K} \sum_{k=1}^K I\[\hat(y)_k=c\], c \in [C]$$

Use the pseudo-labels to approximate the clean class psoterior probabilities

Denote the set of confident examples beloning to the c-th class as $\tau_c$

equation (3)

where l refers to cross-entropy loss and $\gamma_c$ is a threshold for c-th class. $\gamma_c$ is set in a way to get a class-balanced set of confident examples.

The confident example set for all classes is then defined as the union of $\tau = \{\tau_c\_{c=1}^C$


**How to select confident pairs?**


The confident examples are transformed into a set of confident pairs as the union of two different sets. The first set is defined as shown below:

equation (4)

Where P_ij is the pair built by the examples $(x_i,\tilde{y}_i)$ and $(x_j, \tilde{y}_j)$. This set consists of all possible pairs of examples from tau with the same label. The second set is defined on the whole training dataset as:

equation (5)

Where $\tilde{s}_{ij} = I[\tilde{y}_i, \tilde{y}_j]$ and gamma is a dynamic threshold to control the number of identified condiferent pairs. This set represents examples that are misclassified to the same class. The final set of confidents pairs is defined as:

$g = g' \cup g''$

**How is the network trained?**






