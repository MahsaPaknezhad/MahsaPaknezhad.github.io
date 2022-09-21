---
title: 'Selective Supervised Contrastive Learning'
date: 2022-06-10
permalink: /posts/2022/06/selective-supervised-contrastive/
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

Contrastive Learning is able to learn good latent representations that can be used to achieve high performance in downstream tasks. Supervised contrastive learning enhances the learned representations using supervised information. However, noisy supervised information corrupts the learned representations. In this blog post, I will summarize the paper published in CVPR 2022 that proposes an algorithm to learn high quality representations in existence of noisy supervised information. The title of this paper is **Selective Supervised Contrastive Learning with Noisy Labels**. 


A brief on contrastive learning
------
Contrastive learning (Con) aims to learn representations in a latent space that are closer to each other for similar examples and are far from each other for dissimilar examples [1]. The advantage of Con algorithms is that by first learning good latent representations from an unlabeled dataset, they can learn a downstream task with high performance using a small labeled portion of the dataset. While training the Con network, pairs of examples are fed to the network. Originally, only one similar pair (positive pair) and one dissimilar pair (negative pair) were fed to the network. Recently, the proposed algorithms feed the network with a batch that includes multiple positive and negative pairs.  











