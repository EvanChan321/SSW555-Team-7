# BCI Project


## Data Preprocessing (Matlab code)

You can download it from this Google Drive link: [https://drive.google.com/drive/folders/1ykR-mn4d4KfFeeNrfR6UdtebsNRY8PU2?usp=sharing]. 
Please download the data and place it in your data_path at "./data."

# Getting Started
  * You can run the code on your Colab or Kaggle (If your model node need GPU)
    * [1.BCI Data Description](https://colab.research.google.com/drive/1gwIGFoSsRLu9-X3Z79xrPMKiqqE_XeMu#scrollTo=366c60d7)
    * [2.BCI EEG classification](https://colab.research.google.com/drive/1E4kb9iRZHc251SXFANzdpfLBfWzw9L2G)

# Other EEG featuers extraction methods
- [Matlab packages](#Matlab packages)

## Method

* Classification (Machine learning methods, Scikit-learn[https://scikit-learn.org/stable/])
* Classification (Simple Autoencoder model)

## Data introduction & papers
[2020 BCI competition](https://www.frontiersin.org/articles/10.3389/fnhum.2022.898300/full):  – raw data downloaded and tried D2: data set C(Imagined speech) 


## Model Performance:

Name | Epochs | ACC | Precision | Recall | 
---  |:---------:|:---------:|:---------:|:---------:
Transformer | 500 | Best run:54.69% | - | - | -
DNN | 300 | 53.81% | - | - | -
Classification | - |  ~40% for subject, ~ 35% for cross subject | - | - | -
Confomer | - |  - | - | - | -
RNN | - | ~25% for cross subject  | - | - | -


## References:
- [2020 International brain–computer interface competition: A review](https://www.frontiersin.org/articles/10.3389/fnhum.2022.898300/full)