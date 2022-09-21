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
Since the term was first coined in $2012$, Radiomics has widely been used for medical image analysis. Radiomics refers to the automatic or semi-automatic extraction of a large of number of quantitative features from medical images. These features have been able to uncover characteristics that can differentiate tumoral tissue from normal tissue and tissue at different stages of cancer.

In this blog, we aim to show that Radiomic features can be useful for analysis of images in many other domains. The example we have provided here shows that radiomic features can be used for tree bark identification. We use **Trunk12**, a publicly available dataset of tree bark images from [here](https://www.vicos.si/resources/trunk12/).

## Trunk12 Dataset

**Trunk12** consists of $393$ RGB images of tree barks captured from $12$ types of trees that can be found in Slovenia. For each type of tree, there exists about $30$ jpeg images of resolution $3000 \times 4000$ pixels. The images are taken using the same camera Nikon COOLPIX S3000 and while following the same imaging setup: same distance, light conditions and in an upright position. The number of images in each class are shown below.

<p align="center">
<img src="/images/num_of_images_per_class.png" width=800>
</p> 

A few examples of images in this dataset together with their tree type are shown in the figure below. 

<p align="center">
<img src="/images/image_examples_original.png" width=850>
</p> 

To prepare this dataset for radiomic feature extraction we performed a few preprocessing steps on the images. These steps are explained in the following section. 

## Preprocessing

All images went throught the following preprocessing steps. First, the images were converted to grayscale images. Second, squares of size $3000 \times 3000$ pixels were cropped from the center of images. Third, the cropped squares were downsampled to the size $250 \times 250$ pixels. Finally, image contrast was increased so that the intensity values in each image covered the range $[0,255]$. Below, we show the same images that were shown above after going through these preprocessing steps. 

<p align="center">
<img src="/images/image_examples_processed.png" width=850>
</p> 

In the next step, we will extract Radiomic features from the processed images. But first we will provide a brief introduction on Radiomic features. 

## What are Radiomic Features?
Radiomics is quantifying and extracting many imaging patterns including texture and shape features from images using automatic and semi-automatic algorithms. ​These features, usually invisible to the human-eye, can be extracted non-subjectively and used to train and validate models for prediction and early stratification of patients. Radiomic features are categorized into five group of features.

<p align="center">
<img src="/images/radiomics-features.png" width=550>
</p> 
   
These features are extracted from the region of interest in an image which is specified by a mask. We will give a brief explanation for each family of features:

**Histogram-based features**: These are the group of quantitative features that can be extracted from the histogram of the intensity values for the region of interest in the image. Examples of these features include *mean, max and min* intensity values, *variance, skewness, entropy* and *kurtosis*. 

**Transform-based features**: Tranform-based features are features that are extracted after transfering the region of interest to a different space by applying a transformation function. Such transformation functions include *Fourier, Gabor* and *Harr wavelet* transform. Quantitative features such as histogram-based features are extracted from the transformed region of interest.

**Texture-based features**: These features aim to extract quantitative features that represent variations in the intensities within the region of interest. Examples of features in this family of features are:

<p align="center">
<img src="/images/texture-based-features.png" width=600>
</p> 

Extracting any of these features first requires specifying the direction of extraction. Any of the following directions can be selected:

<p align="center">
<img src="/images/direction.png" width=500>
</p> 

As an example, extracting GLCM from a region of interest outputs a matrix. Elements of this matrix specify the number of times different combination of intensity values occur in the region of interest in that direction.
 
<p align="center">
<img src="/images/glcm.png" width=700>
</p> 

As can be seen, GLCM is not a quantitative feature per-se but quantitative features are extracted from GLCM. Some of the quantitative features that can be extracted from GLCM are shown in the table below: 

|<span style="display: inline-block; width:100px">Texture Matrix</span>| Features | Description|
|:-------------- | :-------- |:-------- |
|GLCM | Contrast | Measures the **local variations** in the GLCM.|
| | Correlation | Measures the **joint probability occurrence** of the specified pixel pairs.|
| | Energy | Provides the sum of squared elements in the GLCM. Also known as **uniformity** or the angular second moment.|
| | Homogeneity |Measures the **closeness of the distribution of elements** in the GLCM to the GLCM diagonal.|

**Model-based features**: Parameterised models such as autoregressive models or fractal analysis models can be fitted on the region of interest. Once the parameters for these models are estimated, they are used as Radiomics features. 

**Shape-based features**: Shape-based features describe geometric properties of the region of interest. *Compactness, sphericity, density, 2D or 3D dimeters, axes and their ratios* are examples of features in this family. 

Now that we have a better idea what Radiomics features are, we will proceed with extracting these features from our processed tree bark images. 

## Radiomic Feature Extraction
To extract Radiomics features from our dataset of tree bark images we take advantage of the [PyRadiomics](https://pyradiomics.readthedocs.io/en/latest/) library. This library 



<p align="center">
<img src="/images/example_radiomics.png" width=800>
</p> 















