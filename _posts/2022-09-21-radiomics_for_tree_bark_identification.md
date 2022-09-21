---
title: 'Using Radiomics for Tree Bark Identification'
date: 2022-09-21
permalink: /posts/2022/09/using-radiomics-for-tree-bark-identification/
header-includes:
   - \usepackage{amssymb}
   - \usepackage{amsmath}
output:
    pdf_document
tags:
  - Radiomics Analysis
  - Classification
  - Feature Extraction
---

Contrastive Learning is able to learn good latent representations that can be used to achieve high performance in downstream tasks. Supervised contrastive learning enhances the learned representations using supervised information. However, noisy supervised information corrupts the learned representations. In this blog post, I will summarize the paper published in CVPR 2022 that proposes an algorithm to learn high quality representations in existence of noisy supervised information. The title of this paper is **Selective Supervised Contrastive Learning with Noisy Labels**. 

A brief on contrastive learning
------
<p align="center">
<img src="/images/num_of_images_per_class.png" width=800>
</p> 

<p align="center">
<img src="/images/image_examples_original.png" width=800>
</p> 

<p align="center">
<img src="/images/image_examples_processed.png" width=800>
</p> 

<p align="center">
<img src="/images/example_radiomics.png" width=800>
</p> 













