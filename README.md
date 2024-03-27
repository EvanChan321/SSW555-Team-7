# Lie Detection Website
This is the code for Team 7 (Team Motion). Our goal is to create a lie detection website. Users will upload EEG data as a .npy file and the website will use a machine learning algorithm to determine whether someone is lying based on the data that was uploaded. The website will display the result along with a confidence index of its evaluation. 
 

# Frontend
  ## Running the Frontend 
  ```
  cd client
  npm run dev
  ```

# Backend
  * _Backend Team add stuff here_

# Machine Learning Model 
## Method

* Classification (Machine learning methods, Scikit-learn[https://scikit-learn.org/stable/])
* Classification (Simple Autoencoder model)

## Data introduction & papers
[2020 BCI competition](https://www.frontiersin.org/articles/10.3389/fnhum.2022.898300/full):  – raw data downloaded and tried D2: data set C(Imagined speech) 


## Model Performance:

Name | Epochs | ACC | Precision | Recall | 
---  |:---------:|:---------:|:---------:|:---------:
Transformer | 500 | Best run: 54.69% | - | - | -
DNN | 300 | 53.81% | - | - | -
Classification | 50 |  Best run: 50.99% | - | - | -
Confomer | - |  - | - | - | -
RNN | - | ~25% for cross subject  | - | - | -

## References:
- [2020 International brain–computer interface competition: A review](https://www.frontiersin.org/articles/10.3389/fnhum.2022.898300/full)