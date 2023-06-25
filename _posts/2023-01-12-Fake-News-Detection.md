---
title: "Fake News Detection"
date: 2023-01-12
mathjax: true
toc: true
categories:
  - blog
tags:
  - Numpy
  - project
---

The problem at hand is called fake news detection. In the context of information technology and AI, "fake news" is misleading or false information presented as true news. With the rise of social media and online platforms, the spread of fake news has been prevalent. It can be harmful in many ways, like influencing public opinion based on false information or causing unnecessary panic and confusion among people.

The AI technique used to solve this problem falls under the domain of Natural Language Processing (NLP) which is a subfield of artificial intelligence that focuses on the interaction between computers and humans using natural language. The goal of NLP is to read, decipher, understand, and make sense of human language in a valuable way.

Specifically, the machine learning model used here is called the PassiveAggressiveClassifier. This is a type of online learning algorithm. The online learning model is very suitable for large scale learning problems, and it's quite useful when we have a large stream of incoming data, where it's not feasible to train over the entire data set.

The PassiveAggressiveClassifier is part of a family of algorithms for large-scale learning. It's very similar to the Perceptron in that it does not require a learning rate. However, it does include a regularization parameter.

In layman terms, this is an algorithm that remains 'passive' when dealing with an outcome that has been correctly classified but turns 'aggressive' in the event of a miscalculation, updating and adjusting itself to avoid the mistake in the future.

### The specific tasks it's used for here include:

Text Feature Extraction: Before we feed the text into a machine learning model, we have to convert it into some kind of numeric representation that the model can understand. This is where CountVectorizer comes in. It's a method used to convert the text data into a matrix of token counts.

Text Classification: This is the task of predicting the class (i.e., category) of a given piece of text. Here, we use it to predict whether a given piece of news is "real" or "fake". The PassiveAggressiveClassifier is particularly well-suited to this task because it can efficiently handle large amounts of data and provide accurate predictions.

### Step 1: Import Necessary Libraries



```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
```

### Step 2: Load the Data


```python
# Read the data
df = pd.read_csv('train.csv')
```

### Step 3: Inspect the Data


```python
# Display the first few records
print(df.head())

# Summary of the dataset
print(df.info())
```

       id                                              title              author  \
    0   0  House Dem Aide: We Didn’t Even See Comey’s Let...       Darrell Lucus   
    1   1  FLYNN: Hillary Clinton, Big Woman on Campus - ...     Daniel J. Flynn   
    2   2                  Why the Truth Might Get You Fired  Consortiumnews.com   
    3   3  15 Civilians Killed In Single US Airstrike Hav...     Jessica Purkiss   
    4   4  Iranian woman jailed for fictional unpublished...      Howard Portnoy   
    
                                                    text  label  
    0  House Dem Aide: We Didn’t Even See Comey’s Let...      1  
    1  Ever get the feeling your life circles the rou...      0  
    2  Why the Truth Might Get You Fired October 29, ...      1  
    3  Videos 15 Civilians Killed In Single US Airstr...      1  
    4  Print \nAn Iranian woman has been sentenced to...      1  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20800 entries, 0 to 20799
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   id      20800 non-null  int64 
     1   title   20242 non-null  object
     2   author  18843 non-null  object
     3   text    20761 non-null  object
     4   label   20800 non-null  int64 
    dtypes: int64(2), object(3)
    memory usage: 812.6+ KB
    None


Before we begin pre-processing, we are inspecting our data. This gives us a rough idea about the dataset's structure and any potential issues it might have such as missing values.

### Step 4: Prepare the Labels


```python
# Get the labels
labels = df.label
```

### Step 5: Split the Data


```python
# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)
```

We split our dataset into a training set and a test set. This is to ensure that we have a fair evaluation of our model, by testing it on unseen data.

### Step 6: Handle Missing Values


```python
# Fill NaN values with empty string
x_train = x_train.fillna('')
x_test = x_test.fillna('')
```

We're handling any potential missing values in our dataset. Since our feature is text, we can fill missing values with an empty string.

### Step 7: Initialize and Apply Count Vectorizer


```python
# Initialize a CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data 
count_train = count_vectorizer.fit_transform(x_train.values)
# Transform the test data
count_test = count_vectorizer.transform(x_test.values)
```

We're initializing our CountVectorizer and fitting it to our data. This converts our text data into a format that our model can understand.

### Step 8: Train the Model


```python
# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(count_train, y_train)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>PassiveAggressiveClassifier(max_iter=50)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">PassiveAggressiveClassifier</label><div class="sk-toggleable__content"><pre>PassiveAggressiveClassifier(max_iter=50)</pre></div></div></div></div></div>



Here we're initializing our PassiveAggressiveClassifier and fitting it to our training data.

### Step 9: Make Predictions and Evaluate the Model


```python
# Predict on the test set and calculate accuracy
y_pred = pac.predict(count_test)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Confusion matrix
confusion_mat = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', confusion_mat)
```

    Accuracy: 94.18%
    Confusion Matrix:
     [[1930  130]
     [ 112 1988]]


We are making predictions on our test set and evaluating our model's performance. In this case, we're using accuracy as our metric.

### Step 10: Visualize Results


```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get the confusion matrix
cm = confusion_matrix(y_test,y_pred)

# Plot the confusion matrix in a heat map
plt.figure(figsize=(7,7))
sns.heatmap(cm, annot=True, fmt="d")
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')

```




    Text(58.222222222222214, 0.5, 'True')




    
<img src="https://github.com/DLevin02/DLevin02.github.io/blob/main/assets/images/FakeNews1.png?raw=true" />        



```python
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# We'll use CountVectorizer to count the word frequencies
vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
train_matrix = vectorizer.fit_transform(x_train)

# Get the word frequencies
word_freq_df = pd.DataFrame(train_matrix.toarray(), columns=vectorizer.get_feature_names_out())
word_freq = word_freq_df.sum(axis=0)

# Get the 20 most common words
top_words = word_freq.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 8))
sns.barplot(x=top_words.values, y=top_words.index)
plt.title('Top 20 words in fake news texts')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()

```


    
<img src="https://github.com/DLevin02/DLevin02.github.io/blob/main/assets/images/FakeNews2.png?raw=true" />         

