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
You can do this step within the `sklearn` pipeline ([example](https://stackoverflow.com/questions/67956414/from-train-test-split-to-cross-validation-in-sklearn-using-pipeline)), but I chose to do it outside so each model I'm comparing uses the same dataset. Regardless of what you decide, you must have your train and test sets separate to avoid [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)).

As mentioned above, since the data is imbalanced I'll split the data proportionally so that my test and train data have the same proportions of negative and positive classes.

```r
def split_train_test(df:pd.DataFrame):
    ''' 
    This function takes a df, applies stratified k fold which proportionally splits the data to train and test data

    Args:
        dataframe(df): pd dataframe
    Returns:
        train_data(df): train dataframe
        test_data(df): test dataframe
    '''

    #stratified k fold
    skf = StratifiedKFold(n_splits = 5, 
                          random_state = 11,
                          shuffle = True
                          )
    X = df["text"] # collection of text
    y = df["class"] # class we want to predict

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #combine data  
    train_data = pd.concat([X_train,y_train], axis = 1)
    train_data = train_data.reset_index(drop=True)

    test_data = pd.concat([X_test,y_test], axis=1)
    test_data = test_data.reset_index(drop=True)

    return train_data, test_data

#apply data split function
train_data, test_data = split_train_test(data)
```
<img width="220" alt="image" src="https://github.com/aelb66/News-Sentiment-Classifier/assets/75398560/1d262876-2cc2-46be-b760-e110a1dd725a">


## 4. Pre-processing
I cleaned the data by removing extra whitespaces, converting all words to lowercase, expanding contractions, etc. I then apply the function to train and test set.

```r
ef clean(df:pd.DataFrame):
    ''' 
    This function takes a dataframe, takes the text column to removes extra whitespaces,blank rows and resets index,changes all text to lower case, 
    uses encode function to remove non ASCII characters, uses contractions library to expand common English contractions, removes stop words and 
    converts to lemmatized text.

    Args:
        dataframe(df): a data frame
    Returns:
        dataframe(df): cleaned and normalised dataframe
    '''
    #remove leading and trailing whitespace
    df = df.apply(lambda x: x.str.strip())
    #convert text to lowercase
    df['text'] = df['text'].apply(lambda x: x.lower())
    #remove non ASCII characters
    df['text'] = df['text'].apply(lambda x: x.encode("ascii",errors="ignore").decode())
    #expand common contractions
    df['text'] = df['text'].apply(lambda x: [contractions.fix(word) for word in x.split()]).apply(lambda x:" ".join(word for word in x))
    #remove any punctuations not included in the below
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\d\s\!\?\-\%\.]+', '', x, flags=re.S))
    #any whitespace occuring more than once
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.M|re.S))
    #any whitespace occuring more than once before the punctuation below
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+([.%!?-])', r'\1', x, flags=re.M|re.S))
    #any whitespace occuring more than once
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x, flags=re.M|re.S))
    #remove leading and trailing whitespace
    df = df.apply(lambda x: x.str.strip())
    #remove blanks and reset the index
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

#chose to lemmatize and remove stop words after augmentation as the augmented text were too similar to original when lemmatising before augmentation

#apply the clean function to train and test data
train_c,test_c = clean(train_data),clean(test_data)
```

### 4.1 Data Augmentation
So the data is imbalanced, with more samples in the positive class. To prevent bias towards the positive class I performed data augmentation to the negative class. This involves upsampling or increasing the samples in the negative to the size of the positive class, so both classes are the same size. Data augmentation is only done on the train data. 

There are many ways you can perform data augmentation, click [here](https://neptune.ai/blog/data-augmentation-in-python) for more information. 

I chose to do word replacement via contextual word embeddings which is commonly used. I also chose BERT, a popular pre-trained transformer model, as the pre-trained embedding. This will essentially look at the words before and after the word we want to replace, using context, it then gets what it thinks is the best synonym to replace our word. 

From the below code, we can see that we need to increase the negative class by ~606 samples to be the same proportion as the positive class. 

```r
#size of majority class
maj_class_size = train_c["class"].value_counts().max()
#size of minority class
min_class_size = train_c["class"].value_counts().min()
#the difference between class sizes
diff_class_size = maj_class_size - min_class_size
print(diff_class_size)
```
<img width="152" alt="image" src="https://github.com/aelb66/News-Sentiment-Classifier/assets/75398560/c9c7cbef-0004-4189-bbca-ec1a7d4973eb">

I first initialise the pre-trained word embedding using the `nlpaug` library. I then separate the positive and negative classes using the train data. I initialise an empty list called `aug_texts`, this is where the augmented data will be stored. The code then loops through 606 rows in the negative class (cycles through the negative class twice as it only contains 484 samples) and generates new text data from the negative class by replacing random words with their synonyms. The augmented text data is then appended to `aug_texts`.

```r
#using synonym replacement via contextual word embeddings
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

#separate the minority and majority rows into 2 dfs
min_class_df = train_c[train_c['class'] == "negative"]
maj_class_df = train_c[train_c['class'] == "positive"]

#initialise the augmented text list
aug_texts = []
#for each row up to the size of the difference, augment the negative class and cycle through it till it augments up to the difference
# and append to the augmented text list
for i in range(diff_class_size):
    augmented_text = aug.augment(min_class_df.iloc[i % min_class_size]['text'])
    aug_texts.append(augmented_text)
```

I then create a new data frame called `augmented_train` which contains the augmented text. This is then concatenated to the data frame containing only the negative class. Then lastly I combined that with the data frame containing the positive class. 

```r
#create df from augmented data
augmented_train = pd.DataFrame({'text': aug_texts, 'class': 'negative'})
#convert list to str
augmented_train['text'] = augmented_train['text'].astype(str)
#remove ['']
augmented_train['text'] = augmented_train['text'].str.strip("[]''")
#concatenate negative df with augmented df
combined_train = pd.concat([min_class_df, augmented_train])
#concatenate this modified negative df with positive df
train_caug = pd.concat([combined_train, maj_class_df])
train_caug = train_caug.reset_index(drop=True)
```

### 4.2 Lemmatisation and Stop Word Removal


# CODE

 
## Concluding Notes, Further Exploration and References
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
