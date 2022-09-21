---
title: 'Using Radiomics for Tree Bark Identification'
date: 2022-09-20
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

Using Radiomics for Tree Bark Identification
------
Since the term was first coined in 2012, Radiomics has widely been used for medical image analysis. Radiomics refers to the automatic extraction of a large of number of features from medical images. These features have been able to uncover characteristics that can differentiate tumoral tissue from normal tissue and tissue at different stages of cancer. In this blog, we aim to show that Radiomic features can be useful for analysis of images in many other domains. The example we provide in this blog shows that radiomic features can be used for tree bark identification. We use *Trunk12*, a publicly available dataset of tree bark images from [here](https://www.vicos.si/resources/trunk12/).


*Trukn12* consists of 393 images of tree barks of 12 different trees that can be found in Slovenia. For each tree class, there exists about 30 jpeg images of resolution $3000 \times 4000$ pixels. The images are taken using the same camera Nikon COOLPIX S3000 following the same imaging setup: same distance, light conditions and in an upright position. 

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















