# -*- coding: utf-8 -*-
"""creditCardFrud.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aA6Vq_xTMbJgYDqgY44mIkCVfagBF-kY
"""

# !pip install kaggle

# !mkdir -p ~/.kaggle
# !mv kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json

# !kaggle datasets download -d mlg-ulb/creditcardfraud

# !unzip creditcardfraud.zip

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, mutual_info_classif

from sklearn.metrics import  confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import TargetEncoder,
# Make NumPy printouts easier to read.
np.set_printoptions(precision=5, suppress=True)

df=pd.read_csv('creditcard.csv')
df.head()

df.info()

df.isna().sum()

true_df=df[df.Class==1]
true_df.shape

probability=(df.Class == 0).astype(int)
probability=probability/probability.sum()
probability.head()

np.random.seed(42)

N,temp=df.shape

samples_index = np.random.choice(df.index, size=20000, replace=False, p=probability)

df_sampled = df.loc[samples_index]

df_combined = pd.concat([df_sampled, true_df])
df_combined.shape

df= df_combined

X_train, X_val, y_train, y_val = train_test_split(df.index.values,
                                                  df.Class.values,
                                                  test_size=0.2,
                                                  random_state=42,
                                                  stratify=df.Class.values)
df.drop(columns=['Class'],inplace=True)
train_df=df.loc[X_train]
test_df = df.loc[X_val]
test_df.head()

# normalization
for col in df.columns:
  mean_value = train_df[col].mean()
  std_value = train_df[col].std()

  train_df[col] = (train_df[col] - mean_value) / std_value
  test_df[col] = (test_df[col]-mean_value)/std_value
train_df.sample(1)

x_train=train_df
y_train

x_test=test_df
y_test=y_val

N=20

selector = SelectKBest(score_func=mutual_info_classif, k='all')
selector.fit(x_train, y_train)

feature_scores = pd.DataFrame({'Feature': x_train.columns, 'Information_Gain': selector.scores_})
feature_scores = feature_scores.sort_values(by='Information_Gain', ascending=False)
selected_features = feature_scores.head(N)['Feature'].tolist()


x_train_subset = x_train[selected_features]
x_test_subset = x_test[selected_features]

x_train_subset.head()

train_x=x_train_subset.values
train_y = y_train

test_x = x_test_subset.values
test_y = y_test

train_y=train_y.reshape(-1, 1)
test_y=test_y.reshape(-1,1)

train_y.shape

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def initialize_weights(num_features):
  return np.zeros((num_features, 1))

def add_intercept_column(X):
  intercept = np.ones((X.shape[0], 1))
  return np.concatenate((intercept, X), axis=1)

def accuracy_with_intercept(X,Y, weights):
  predictions = sigmoid(np.dot(X, weights))
  y_pred=(predictions >= 0.5).astype(int)
  accuracy_ = np.mean(predictions == Y)
  return accuracy_

def logistic_regression(X, y, learning_rate=0.01, num_iterations=5000, error_rate=.5):
  X = add_intercept_column(X)
  m, n = X.shape
  weights = initialize_weights(n)

  for i in range(num_iterations):
    if 1-accuracy_with_intercept(X,y,weights)<error_rate:
      return weights
    z = np.dot(X, weights)
    predictions = sigmoid(z)
    error = predictions - y
    gradient = np.dot(X.T, error) / m
    weights -= learning_rate * gradient

  return weights

print(logistic_regression(train_x,train_y,error_rate=0.0))

def AdaBoost(X,y,l_weak,K,seed=42,error_rate=.5):
  np.random.seed(seed)

  intercepted_X=add_intercept_column(X)
  N,temp=X.shape

  w = np.ones(N) / N
  w = w.reshape(-1, 1)

  h=[]
  weights=np.zeros(K)

  for k in range(K):
    samples_index=np.random.choice(N, size=N, replace=True, p=w.flatten())

    samples_x=X[samples_index]
    samples_y= y[samples_index]

    h_k = l_weak(samples_x,samples_y)

    z = np.dot(intercepted_X, h_k)
    predictions_probability=sigmoid(z)
    predictions_lebel=(predictions_probability >= 0.5).astype(int)
    print((predictions_lebel!=y).astype(int).shape)

    error=np.sum(np.dot(w.T,predictions_lebel!=y))

    print(error)

    if error> .5:
      continue

    for i in range(N):
      if predictions_lebel[i] == y[i]:
        w[i]=w[i]*error/(1-error)
    w=w/np.sum(w)

    h.append(h_k)
    weights[k]=np.log((1 - error) / max(error, 1e-10))
  return h,weights

def weighted_majority(h, z, x):
  votes=np.zeros((x.shape[0],1))

  print(votes.shape)
  x=add_intercept_column(x)
  for h_z, z_k in zip(h,z):

    z_values=np.dot(x,h_z)
    predictions_probability=sigmoid(z_values)
    predictions_lebel=(predictions_probability >= 0.5).astype(int)

    predictions_lebel[predictions_lebel == 0] = -1

    print(predictions_lebel.shape)
    votes+=predictions_lebel
  return (votes>0 ).astype(int)

h,z=AdaBoost(train_x,train_y,logistic_regression,20, seed=42)

y_pred=weighted_majority(h,z,test_x)

conf_matrix = confusion_matrix(test_y.flatten(), y_pred.flatten())
TN, FP, FN, TP = conf_matrix.ravel()

# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# True Positive Rate (Recall)
recall = TP / (TP + FN)

# True Negative Rate (Specificity)
specificity = TN / (TN + FP)

# Positive Predictive Value (Precision)
precision = TP / (TP + FP)

# False Discovery Rate
fdr = FP / (TP + FP)

# F1 Score
f1_score = 2 * (precision * recall) / (precision + recall)

# Print the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall (True Positive Rate): {recall:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")
print(f"Precision (Positive Predictive Value): {precision:.4f}")
print(f"False Discovery Rate: {fdr:.4f}")
print(f"F1 Score: {f1_score:.4f}")

