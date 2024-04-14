---
title: 'Enhancing Geological Mapping through Inpainting of Surface Infrastructure Artefacts '
date: 2024-04-12
permalink: /posts/2024/04/inpainting-surface-geological-artefacts/
header-includes:
   - \usepackage{amssymb}
   - \usepackage{amsmath}
output:
    pdf_document
tags:
  - Latent Diffusion Model
  - Inpainting
  - Data Cleaning
  - Sentinel2
---

Remote sensing data such as aerial and satellite imagery serve as valuable tools for mapping geological  features.  However,  these  datasets  often  present  statistical  limitations  by  capturing only the surface of the Earth. This limitation becomes pronounced when regions of interest, such as economic mineralization or specific geological formations are studied mainly because surface infrastructure artefacts, including mines, roads, dams, and drill pads, inadvertently influence the signature  within  the  imagery.  Such  non-geological  artefacts  interfere  with  machine  learning models when focused on geological responses, necessitating the masking or removal of these 
artefacts.

## Enhancing Geological Mapping Through Inpainting of Surface Infrastructure Artefacts 

 
Recent  advancements  in  machine  learning  and  image  processing  demand  comprehensive datasets  devoid  of  missing  data.  To  address  this  challenge,  various  imputation  methods  have been developed for inpainting missing spatial data. Among these methods, diffusion techniques emerge  as  modern,  state-of-the-art  approaches  for  inpainting  data  based  on  surrounding geological context. 

<p align="center">
<img src="/images/Inpainting_diagram.jpg" width=800>
</p> 


In  the  field  of  computer  vision  and  artificial  intelligence  (AI),  existing  inpainting  models, including diffusion models, are predominantly designed for processing natural images composed of  the  standard  red,  green,  and  blue  (RGB)  colour  channels.  However,  in  remote  sensing applications, image  data often  consists of  more  than just the three  RGB bands.  This expanded dataset  may  include  numerous  spectral  bands,  each  providing  valuable  information  about the geological features present. 

## Multi-channel Inpainting Model for Data Cleaning

Traditionally, one approach to address this complexity involves treating each spectral band as a  separate  greyscale  image  and  training  individual  inpainting  models  for  each  band.  While feasible, this method overlooks the potential benefits of leveraging the inter-band relationships inherent  in  multi-band  imagery.  A  more  effective  strategy  entails  training  a  single  inpainting model capable of processing all bands simultaneously. By doing so, the model can better exploit the intricate relationships between spectral bands to inpaint missing regions in a more realistic manner. 

Despite the advantages of this unified approach, existing AI frameworks often lack support for images with more than three bands, limiting the applicability of traditional inpainting methods in geological  contexts.  To  address  this  gap,  we  have  developed  an  advanced  inpainting  model specifically tailored to handle images with an arbitrary number of spectral bands. Our model not only  overcomes  these  technical  limitations  but  also  enhances  the  accuracy  and  versatility  of inpainting tasks in geological imagery analysis. 

This  porject  underscores  the  significance  of  inpainting  methodologies  in  enhancing  the accuracy and reliability of geological mapping based on aerial and satellite imagery. By 
effectively  removing  surface  infrastructure  artefacts  and  inpainting  missing  spatial  data,  these techniques  enable  more  precise  prospectivity  mapping  and  facilitate  the  advancement  of geological understanding and exploration strategies in diverse landscapes.

Figure above shows how our multi-channel inpainting model is used to clean the roads in the 12-channel multispectral imagery (Sentinel-2) data of North Flinders range. 


## Downstream Tasks after Data Cleaning

One of the widely used AI models in many of our projects is our self-supervised learning model that is custom-built for analyzing geological imagery. By leveraging our innovative multi-channel self-supervised learning approach, we've been able to extract invaluable features. These features play a pivotal role in downstream tasks such as clustering (see Figure below), which enables the identification of similar geological features or creation of similarity maps that point out regions that have similar attributes as a reference region.


<p align="center">
<img src="/images/downstream_tasks.jpg" width=800>
</p> 

## Conclusion

In this blog, we introduced a multi-channel inpainting model tailored specifically for geological imagery. Our proposed model performs geological imagery data cleaning by inpainting areas containing missing data or surface infrastructure artefacts such as mines, roads, dams, and drill pads.



















