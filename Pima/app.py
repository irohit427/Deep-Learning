# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 20:56:39 2020

@author: rohit
"""

import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
headers = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

data = pd.read_csv(r'C:\Users\irohi\Desktop\Deep Learning\ANN\Pima\data\pima-indians-diabetes.csv', names=headers)

X = data.drop(columns=['Outcome'])
y = data['Outcome'].values.reshape(data.shape[0], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=500, batch_size=50, verbose=1)
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]

print("Train accuracy of keras neural network: {}".format(round((train_acc * 100), 2)))
print("Test accuracy of keras neural network: {}".format(round((test_acc * 100),2)))

