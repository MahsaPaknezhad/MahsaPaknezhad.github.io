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
Since the term was first coined in $2012$, Radiomics has widely been used for medical image analysis. Radiomics refers to the automatic or semi-automatic extraction of a large of number of quantitative features from medical images. These features have been able to uncover characteristics that can differentiate tumoral tissue from normal tissue and tissue at different stages of cancer. In this blog, we aim to show that Radiomic features can be useful for analysis of images in many other domains. The example we provide in this blog shows that radiomic features can be used for tree bark identification. We use *Trunk12*, a publicly available dataset of tree bark images from [here](https://www.vicos.si/resources/trunk12/).


*Trunk12* consists of $393$ RGB images of tree barks of $12$ different trees that can be found in Slovenia. For each type of tree, there exists about $30$ jpeg images of resolution $3000 \times 4000$ pixels. The images are taken using the same camera Nikon COOLPIX S3000 and while following the same imaging setup: same distance, light conditions and in an upright position. The number of images in each class are shown below.

<p align="center">
<img src="/images/num_of_images_per_class.png" width=800>
</p> 

A few examples of images in this dataset together with their tree type are shown in the figure below. 

<p align="center">
<img src="/images/image_examples_original.png" width=800>
</p> 

To prepare this dataset for radiomic feature extraction we performed a few preprocessing steps on the images. These steps are explained in the following section. 

## Preprocessing

All images went throught the following preprocessing steps. First, the images were converted to grayscale images. Second, squares of size $3000 \times 3000$ pixels were cropped from the center of images. Third, the cropped squares were downsampled to the size $250 \times 250$ pixels. Finally, image contrast was increased so that the intensity values in each image covered the range $[0,255]$. Below, we show the same images that were shown above after going through these preprocessing steps. 

<p align="center">
<img src="/images/image_examples_processed.png" width=800>
</p> 

In the next step, we will extract Radiomic features from the processed images. But first we will provide a brief introduction on Radiomic features. 

## What are Radiomic Features?
Radiomics is quantifying and extracting many imaging patterns including texture and shape features from images using automatic and semi-automatic algorithms. ​These features, usually invisible to the human-eye, can be extracted non-subjectively and used to train and validate models for prediction and early stratification of patients. Radiomic features are categorized into 5 group of features

<p align="center">
<img src="/images/radiomics-features.png" width=800>
</p> 
   
We will give a brief explanation for each category:

* Histogram-based features: These are the group of quantitative features that can be extracted from the histogram of the intensity values for the region of interest in the image. Examples of these features include mean, maximum and minimum intensity values, variance, skewness, entropy and kurtosis. 

* Texture-based features: 
<p align="center">
<img src="/images/example_radiomics.png" width=800>
</p> 















