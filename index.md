# How I created a sentiment classification model on news data using Python and how you can too

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

## Background and Data 
The purpose of this project was to perform supervised binary sentiment classification on labelled financial news headlines, where a news headline is either positive or negative.

The data is obtained from Kaggle. It can be downloaded [here](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news).

Below, I go through my code step-by-step and in detail so if you're TLDR type of person and just want the code, I'll only be a little butt hurt - visit my github [here](https://github.com/aelb66/News-Sentiment-Classifier).

*hawt tip:* If you're not sure what a specific line/chunk of code does, copy and paste it into [Bing Chat](https://www.microsoft.com/en-us/edge/features/bing-chat?form=MT00D8) and ask it to explain it to you.

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

## 2. Exploratory Data Analysis (EDA)
Above all else, the first thing you should do with data is look at what you're working with. 

Below, I first resized the output display so that you can see the full text of each headline clearly.
I then created a function that imported the data, performed basic cleaning/filtering and returned the cleaned data as a Pandas DataFrame. The original file has 3 classes: positive, negative and neutral. I removed the neutral class out of preference.

```r
#resizing output display
pd.set_option('max_colwidth', None)

def import_clean_df(path:str):
    ''' 
    Takes a path to the original data, reads data as Pandas (pd) DataFrame (df) and does initial cleaning:
    (names columns, drops duplicate and blank rows, resets index) and returns the df

    Args:
        path(str): path to raw data
    Returns:
        dataframe(df): cleaned pd df
    ''' 
    #reading in data and naming columns to class and text
    raw = pd.read_csv(path,skipinitialspace=True, skip_blank_lines=True,encoding = "ISO-8859-1", names=["class","text"]) 
    #subset data to only positive and negative classes
    raw_pn = raw[raw["class"].isin(["positive","negative"])]
    #drop blank rows
    raw_pn = raw_pn.dropna()
    #drop duplicate rows
    raw_pn = raw_pn.drop_duplicates()
    #reset index
    raw_pn = raw_pn.reset_index(drop=True)
    return raw_pn

#path to data
filepath = "all-data.csv"

data = import_clean_df(filepath)

#preview of data
data
```
This is what the data looks like so far. As expected it's quite messy e.g., (spaced-out commas and percentages, whole words that are either capitalised, lowercase or a mix), so further cleaning might be required.

<img width="780" alt="image" src="https://github.com/aelb66/News-Sentiment-Classifier/assets/75398560/f5fef834-a1f5-42d3-8c08-cb291e99853a">

Checking out the frequency and average word counts between classes.
```r
#class frequency
print("Class Frequency:\n",data["class"].value_counts())
#word count per class
data["word_count"] = data["text"].map(lambda x: len(x.split()))
print("\nAverage word count per class:\n",data.groupby("class")["word_count"].mean())
#delete created column after use
del data["word_count"]
```
<img width="245" alt="image" src="https://github.com/aelb66/News-Sentiment-Classifier/assets/75398560/7df22228-410c-486b-887a-0b1293844bd9">

### Initial Observations and Actions
#### Observations:
- Looking at the frequencies, the classes are moderately imbalanced ~(70:30) which can lead to model bias toward the positive class (majority) during training and inference.
- Data is on the smaller side for train/test/validation split.
#### Actions:
- As data is unbalanced, I'll split data proportionally into train/test using stratified k-fold cross-validation.
- To increase the negative sentiment class and to make the dataset more balanced, I will perform data augmentation via word embeddings as shown effective in literature [1].
- For model selection, I will experiment with the following models:
  - Binary Logistic Regression: baseline ML model
  - Random Forest: better at handling imbalanced data
  - DistilRoberta: small, faster version of RoBERTa, a well-known pre-trained transformer model - using the dataset to fine-tune the model

## 3. Split the Data
You can do this step within the sklearn pipeline ([example](https://stackoverflow.com/questions/67956414/from-train-test-split-to-cross-validation-in-sklearn-using-pipeline)), but I chose to do it outside so each model I'm comparing uses the same dataset. Regardless of what you decide, you must have your train and test sets separate to avoid [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)).

As mentioned, since my data is imbalanced I'll split the data proportionally so that my test and train data have the same proportions of negative and positive classes.


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
