# How I created news sentiment classifiers using Python and how you can too

![image of newspapers on a stand](markus-spiske-2G8mnFvH8xk-unsplash.jpg) 
<br />
Photo by <a href="https://unsplash.com/@markusspiske?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Markus Spiske</a> on <a href="https://unsplash.com/photos/2G8mnFvH8xk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

<br />

Check out my [personal website](https://www.alolelba.com/) for more projects. 

<br />


# Contents
{:.no_toc}
* Data 
{:toc}

## Data 

## 1. Libraries 
I used the following libraries for this project. 
```r
# Data pre-processing
import numpy as np
import pandas as pd
import re
import spacy
import contractions
import nlpaug.augmenter.word as naw
from sklearn.preprocessing import LabelEncoder

# Machine Learning pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

# Deep Learning pipeline
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, Trainer, TrainingArguments,EarlyStoppingCallback
import torch

# Metrics
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, balanced_accuracy_score,log_loss
```

## 2. Pre-processing

### 2.1 Extracting data


### Initial Observations and Actions
#### Observations:
- Classes are moderately imbalanced ~(70:30) which can lead to model bias toward positive class (majority) during training and inference.
- Data is on the smaller side for train/test/validation split.
#### Actions:
- As data is unbalanced, I'll split data proportionally into train/test using stratified k fold cv.
- To increase the negative sentiment class and to make the dataset more balanced, I will perform data augmentation via word embeddings as it is shown effective in literature [1].
- For model selection, I will experiment with the following models:
  - Binary Logistic Regression: baseline ML model\n",
  - Random Forest: better at handling imbalanced data\n",
  - DistilRoberta: small, faster version of RoBERTa, a well known pre-trained transformer model - using the dataset to fine-tune the model


## CODE

 
### Concluding Notes, Further Exploration and References
#### Concluding Notes
- DistilRoBERTa (untuned) was found to have the best performance on the data (original: micro_F1 0.96, augmented: macro_F1 0.92), with the
- Random Forest model coming in second [untuned](original: micro_F1 0.87, augmented: macro_F1 0.83) and lastly the baseline model Logistic Regression [untuned](original: micro_F1 0.84, augmented: macro_F1 0.82).
- DistilRoBERTa performed better and produced a more confident model on original data (eval_loss: 0.12, train_loss: 0.17) compared to the cleaned augmented data (eval_loss: 0.23, train_loss: 0.18), this could be due to a number of factors such as:
  - The particular cleaning methods used could have caused information/contextual loss
  - DistilRoBERTa is pre-trained on large,diverse and messy data, so those characteristics could be more closer to the raw data
  - The data augmentation method or parameters may not have been effective
  - DistilRoBERTa like other transformers, in many cases are robust to class imbalance
- It is important to note that these models are limited to English headlines and may not be suitable to headlines in other languages.

#### Further Exploration
- The following can be further explored. These include but are not limited to:
  - Conduct hyperparameter tuning on DistilRoBERTa to optimise the model on the data provided.
  - Perform analysis on more models like NN and LTSM and other ML models like SVM could be explored as it was done in other sentiment classification studies [2][3].
  - Experiment with word embeddings for text representation such as GloVe and BERT which is shown to help with performance [4].
  - Experiment more with different data augmentation techniques and word embedding parameters.
  - Utilise hyperparameter tuning to find the number of k for k-fold cross validation.
  - Perform ensembling with a combination of deep learning and/or machine learning models for performance improvement [5].
  - Increasing sample size by adding other relevant datasets.
 

### References
- [1] https://arxiv.org/abs/2007.02033
- [2] https://doi.org/10.1016/j.jbi.2020.103539
- [3] https://arxiv.org/abs/2006.03541
- [4] https://arxiv.org/abs/1910.03505
- [5] https://doi.org/10.1016/j.eswa.2021.115819
