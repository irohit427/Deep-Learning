# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:50:10 2020

@author: rohit
"""

import pandas as pd

automobile_data = pd.read_csv(r'C:\Users\irohi\Desktop\Deep Learning\ANN\Automobile\data\automobile_data.csv', sep=r'\s*,\s*', engine='python')
import numpy as np
pd.options.mode.chained_assignment = None
automobile_data = automobile_data.replace('?', np.nan)
automobile_data = automobile_data.dropna()
col = ['make', 'fuel-type', 'body-style', 'horsepower']
automobile_features = automobile_data[col]
automobile_taget = automobile_data['price']
automobile_features['horsepower'] = pd.to_numeric(automobile_features['horsepower'])
automobile_taget = automobile_taget.astype(float)
automobile_features = pd.get_dummies(automobile_features, columns=['make', 'fuel-type', 'body-style'])

from sklearn import preprocessing
automobile_features[['horsepower']] =preprocessing.scale(automobile_features[['horsepower']])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(automobile_features, automobile_taget, test_size=0.2, random_state=0)

import torch

dtype = torch.float

X_train_tensor = torch.tensor(X_train.values, dtype=dtype)
X_test_tensor = torch.tensor(X_test.values, dtype=dtype)

y_train_tensor = torch.tensor(y_train.values, dtype=dtype)
y_test_tensor = torch.tensor(y_test.values, dtype=dtype)

input = 26
output = 1

hidden = 100
loss_func = torch.nn.MSELoss()
learning_rate = 1e-4

model = torch.nn.Sequential(torch.nn.Linear(input, hidden),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(hidden, output))

for i in range(10000):
  y_pred = model(X_train_tensor)
  loss = loss_func(y_pred, y_train_tensor)
  
  if i % 1000 == 0:
    print(i, loss.item())
    
  model.zero_grad()
  loss.backward()
  
  with torch.no_grad():
    for param in model.parameters():
      param -= learning_rate * param.grad

y_pred_tensor = model(X_test_tensor)
y_pred = y_pred_tensor.detach().numpy()

import matplotlib.pyplot as plt

plt.figure(figsize=(15,6))

plt.plot(y_pred, label='Predicted Price')
plt.plot(y_test.values, label='Actual Price')
plt.legend()
plt.show()


