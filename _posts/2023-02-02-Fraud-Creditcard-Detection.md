---
title: "Credit Card Fraud Detection Using Random Forest"
date: 2023-02-02
mathjax: true
toc: true
categories:
  - blog
tags:
  - Numpy
  - project
---


## Introduction:

Credit card fraud is a significant concern in today's digital world, with substantial financial and security implications for individuals and financial institutions.
Machine learning techniques provide a promising solution to detect fraudulent transactions and mitigate the risks associated with them.
In this blog post, we explore the application of machine learning algorithms to enhance credit card fraud detection.

## Random Forest

Here's how Random Forest works:

##### Decision Trees:
A decision tree is a flowchart-like structure where each internal node represents a test on a feature, each branch represents the outcome of the test, and each leaf node represents a class label or a decision. Decision trees can be prone to overfitting, meaning they might learn too much from the training data and perform poorly on new, unseen data.

##### Ensemble Learning: 
Random Forest overcomes the limitations of a single decision tree by combining the predictions of multiple decision trees. It creates an ensemble of decision trees and makes predictions by taking a majority vote or averaging the predictions of the individual trees.

##### Random Subsets: 
Random Forest introduces randomness by using random subsets of the original dataset for building each decision tree. This process is known as bagging (bootstrap aggregating). The subsets are created by sampling the data with replacement, meaning that some instances may be repeated in the subsets, while others may be left out. This approach helps to introduce diversity among the trees.

##### Random Feature Selection: 
In addition to using random subsets of data, Random Forest also randomly selects a subset of features at each split of a decision tree. This process helps to decorrelate the trees and ensures that different trees consider different subsets of features. It prevents a single feature from dominating the decision-making process and promotes more robust predictions.

##### Voting or Averaging: 
Once the individual decision trees are built, predictions are made by taking a majority vote (for classification problems) or averaging (for regression problems) of the predictions from each tree. This aggregation of predictions reduces the variance and improves the overall performance of the model.

#### Advantages of Random Forest:
Random Forest is highly accurate and performs well on a wide range of datasets.
It can handle large datasets with high dimensionality.
Random Forest can effectively handle imbalanced datasets and is less prone to overfitting compared to individual decision trees.
It provides a measure of feature importance, allowing for feature selection and interpretation of the model.

<img src="https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png" />    

### Step 1: Import necessary libraries and load the data

The first step in any machine learning problem is to import the necessary libraries and load the data we are going to use.


```python
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the csv file
data = pd.read_csv('creditcard.csv')
```

### Step 2: Data Exploration

Before proceeding to model training, we should first explore our data a bit.


```python
# Understanding the columns
print(data.columns)

# Understanding the shape
print(data.shape)

# Understanding the data types
print(data.dtypes)

# Check for missing values
print(data.isnull().sum())

```

    Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
           'Class'],
          dtype='object')
    (284807, 31)
    Time      float64
    V1        float64
    V2        float64
    V3        float64
    V4        float64
    V5        float64
    V6        float64
    V7        float64
    V8        float64
    V9        float64
    V10       float64
    V11       float64
    V12       float64
    V13       float64
    V14       float64
    V15       float64
    V16       float64
    V17       float64
    V18       float64
    V19       float64
    V20       float64
    V21       float64
    V22       float64
    V23       float64
    V24       float64
    V25       float64
    V26       float64
    V27       float64
    V28       float64
    Amount    float64
    Class       int64
    dtype: object
    Time      0
    V1        0
    V2        0
    V3        0
    V4        0
    V5        0
    V6        0
    V7        0
    V8        0
    V9        0
    V10       0
    V11       0
    V12       0
    V13       0
    V14       0
    V15       0
    V16       0
    V17       0
    V18       0
    V19       0
    V20       0
    V21       0
    V22       0
    V23       0
    V24       0
    V25       0
    V26       0
    V27       0
    V28       0
    Amount    0
    Class     0
    dtype: int64


The above code provides information about the dataset like the column names, the number of records (shape), the data types of the columns, and checks if there are any missing values in the data.

### Step 3: Visualizing the Data

Visualization of data is an essential aspect of any data analysis. Here, we are visualizing the distribution of the classes (fraudulent vs. non-fraudulent transactions)


```python
# Visualizing the classes
count_classes = pd.value_counts(data['Class'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("Transaction class distribution")
plt.xticks(range(2), ["Normal", "Fraud"])
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
```


    
<img src="https://github.com/DLevin02/DLevin02.github.io/blob/main/assets/images/Creditcard3.png?raw=true" />        

    


This code creates a bar chart showing the frequency of fraudulent vs. non-fraudulent transactions. This can be useful in understanding the balance (or imbalance) between the classes in our dataset.

### Step 4: Data Preprocessing

Data preprocessing is an important step in a machine learning project. It transforms raw data into a format that will be more easily and effectively processed for the purpose of the user.

In the given dataset, features are mostly scaled except for Time and Amount. So let's scale these features too.


```python
from sklearn.preprocessing import StandardScaler

# Scaling Time and Amount
scaler = StandardScaler()
data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1,1))

# Dropping old Time and Amount
data = data.drop(['Time','Amount'], axis=1)
```

### Step 5: Splitting the Data into Train and Test Sets

The data is divided into two parts: a training set and a test set. The model is trained on the training data and then tested on the unseen test data to evaluate its performance.


```python
from sklearn.model_selection import train_test_split

# Defining the features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 6: Training the Machine Learning Model

The Random Forest model has a good performance in general because it runs multiple decision trees on various subsets of the dataset and makes the final decision based on the majority voting from all the individual decision trees. It has the ability to limit overfitting without substantially increasing error due to bias.
    
<img src="https://imgopt.infoq.com/fit-in/1200x2400/filters:quality(80)/filters:no_upscale()/articles/fraud-detection-random-forest/en/resources/fraud-detection2-1565620755869.png" />         


```python
# Importing necessary libraries
from sklearn.ensemble import RandomForestClassifier

# Define the model as the random forest
rf_model = RandomForestClassifier(n_estimators=100)

# Train the model
rf_model.fit(X_train, y_train)

# Use the model to make predictions
rf_predicted = rf_model.predict(X_test)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



### Step 7: Evaluating the Model

Now, we will predict the labels for our test set and evaluate the model's performance by comparing these predictions to the actual labels.


```python
from sklearn.metrics import classification_report

# Making predictions on the test data
y_pred = rf_model.predict(X_test)

# Printing the classification report
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00     56864
               1       0.97      0.78      0.86        98
    
        accuracy                           1.00     56962
       macro avg       0.99      0.89      0.93     56962
    weighted avg       1.00      1.00      1.00     56962
    



```python
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Predicting the test set results
y_pred = rf_model.predict(X_test)

# Making the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```


    
<img src="https://github.com/DLevin02/DLevin02.github.io/blob/main/assets/images/Creditcard2.png?raw=true" />        

    



```python
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for predictions on validation set
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

```


    
<img src="https://github.com/DLevin02/DLevin02.github.io/blob/main/assets/images/Creditcard3.png?raw=true" />        

    


## Conclusion

The Random Forest model demonstrates excellent performance in classifying non-fraudulent transactions (Class 0). It achieves perfect precision, recall, and f1-score of 1.00, indicating that it accurately predicts all non-fraudulent transactions without any false positives or false negatives. This suggests that the model is highly reliable in identifying genuine transactions.

For fraudulent transactions (Class 1), the Random Forest model exhibits strong performance as well. It achieves a high precision of 0.97, indicating that out of all transactions predicted as fraudulent, 97% are truly fraudulent. The recall of 0.78 suggests that the model correctly identifies 78% of all actual fraudulent transactions. The f1-score of 0.86 represents a harmonious balance between precision and recall for the fraudulent class.

The overall accuracy of the model is 1.00, which means it correctly classifies transactions in both classes with high accuracy.

In summary, the Random Forest model demonstrates exceptional performance in detecting fraudulent transactions while maintaining a high accuracy rate for non-fraudulent transactions. It achieves a good balance between precision and recall for the fraudulent class, providing a reliable and robust solution for credit card fraud detection.

