---
title: 'Selective Supervised Contrastive Learning with Noisy Labels'
date: 2022-05-10
permalink: /posts/2022/05/selective-supervised-contrastive-learning/
header-includes:
   - \usepackage{amssymb}
   - \usepackage{amsmath}
output:
    pdf_document
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

* Select pairs of examples that the algorithm is confident about their label
* Train the network with the confident pairs using supervised contrastive learning


**How to find confident examples?**

Uses unsupervised training to initially train the network in the first few epochs

Confident examples are found by first measuring cosine distance between the low dimensional representations $z_i$, $z_j$ of each pair of examples

$$d(z_i, z_j) = \frac{z_i z_j^T}{\|z_i\|~\|z_j\|}$$

Creating a pseudo-label $\hat{y}_i$ for each example ($x_i$, $\tilde{y}_i$) by aggregating the original label from its top-k neighbors with lowest the cosine distance

$$\hat{q}_c(x_i) = \frac{1}{K} \sum_{k=1}^K \mathbb{I}[(\hat{y})_k=c], c \in [C]$$

Use the pseudo-labels to approximate the clean class psoterior probabilities

Denote the set of confident examples beloning to the c-th class as $\mathcal{T}_c$

$$ \mathcal{T}_c = \{(x_i, \tilde{y}_i) | \mathcal{l}(\mathbf{\hat{q}}(x_i), \tilde{y}_i) < \gamma_c, i \in [n]\}, c \in [C]$$

where $\mathcal{l}$ refers to cross-entropy loss and $\gamma_c$ is a threshold for c-th class. $\gamma_c$ is set in a way to get a class-balanced set of confident examples.

The confident example set for all classes is then defined as $\mathcal{T} = \bigcup_{c=1}^C \mathcal{T}_c$


**How to select confident pairs?**


The confident examples are transformed into a set of confident pairs as the union of two different sets. The first set is defined as shown below:

$$\mathcal{G}' = \{P_{ij}| \tilde{y}_i = \tilde{y}_j, (x_i, \tilde{y}_i), (x_j, \tilde{y}_j) \in \mathcal{T}\}$$

Where $P_ij$ is the pair built by the examples $(x_i,\tilde{y}_i)$ and $(x_j, \tilde{y}_j)$. This set consists of all possible pairs of examples from $\mathcal{T}$ with the same label. The second set is defined on the whole training dataset as:

$$ \mathcal{G}'' = \{P_{ij} | \tilde{s}_{ij} = 1, d(z_i, z_j) > \gamma\}$$

Where $\tilde{s}_{ij} = \mathbb{I}[\tilde{y}_i, \tilde{y}_j]$ and gamma is a dynamic threshold to control the number of identified condiferent pairs. This set represents examples that are misclassified to the same class. The final set of confidents pairs is defined as:

$$\mathcal{G} = \mathcal{G}' \cup \mathcal{G}''$$

**How is the network trained?**

The network is trained using three loss terms. The first term uses the Mixup technique which generated a convex combination of pairs of examples as $x_i = \lambda x_a + (1-\lambda)x_b$, where $\lambda \in [0,1] \sim Beta(\alpha_m, \alpha_m)$; and $x_a$ and $x_b$ are two mini-batch examples. The Mixup loss is defined as:


$$ \mathcal{L}_i^{MIX} = \lambda \mathcal{L}_a(z_i) + (1-\lambda) \mathcal{L}_b(z_i),$$

where $\mathcal{L}_a$ and $\mathcal{L}_b$ have the same form as supervised contrastive learning loss that is defined as:

$$ \mathcal{L}_i = \sum_{g \in \mathcal{G}(i)} \log \frac{exp(z_i.z_g/\tau)}{\sum_{a\in A(i)} exp(z_i.z_a/\tau)}.$$

$A(i)$ specifies the set of indices excluding $i$ and $\mathcal{G}_i =\{g \in A(i), P_{i'j'} in \mathcal{G}\} $ where $i'$ and $g'$ are the original indices of $x_i$ and $x_g$. Also, $\tau \in \mathbb{R}^+$ is a temperature parameter. 









