# Lie Detection Website
This is the code for Team 7 (Team Motion). Our goal is to create a lie detection website. Users will upload EEG data as a .npy file and the website will use a machine learning algorithm to determine whether someone is lying based on the data that was uploaded. The website will display the result along with a confidence index of its evaluation. 
 
# Running the App 
Make sure you are in the client folder when you run the app.
  ```
  npm run dev
  ```
# Machine Learning Model 

## Method
Model is made using pytorch.
* Classification (Machine learning methods, Scikit-learn[https://scikit-learn.org/stable/])
* Classification (Simple Autoencoder model)

## Data introduction & papers
[2020 BCI competition](https://www.frontiersin.org/articles/10.3389/fnhum.2022.898300/full):  – raw data downloaded and tried D2: data set C(Imagined speech) 


## Model Performance:
Focus on classification

Name | Epochs | ACC | Precision | Recall | 
---  |:---------:|:---------:|:---------:|:---------:
Transformer | 500 | Best run: 54.69% | - | - | -
DNN | 300 | 53.81% | - | - | -
Classification | 50 |  Best run: 51.70% | - | - | -
Confomer | - |  - | - | - | -
RNN | - | ~25% for cross subject  | - | - | -

## References:
- [2020 International brain–computer interface competition: A review](https://www.frontiersin.org/articles/10.3389/fnhum.2022.898300/full)