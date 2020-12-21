# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 18:23:12 2020

@author: rohit
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import NeuralNet

headers =  ['age', 'sex','chest_pain','resting_blood_pressure',  
        'serum_cholestoral', 'fasting_blood_sugar', 'resting_ecg_results',
        'max_heart_rate_achieved', 'exercise_induced_angina', 'oldpeak',"slope of the peak",
        'num_of_major_vessels','thal', 'heart_disease']

heart_df = pd.read_csv(r'C:\Users\irohi\Desktop\Deep Learning\ANN\Heart Disease Prediction\data\heart.dat', sep=' ', names=headers)

X = heart_df.drop(columns=['heart_disease'])
heart_df['heart_disease'] = heart_df['heart_disease'].replace(1, 0)
heart_df['heart_disease'] = heart_df['heart_disease'].replace(2, 1)

y = heart_df['heart_disease'].values.reshape(X.shape[0], 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

nn = NeuralNet.NeuralNet()
nn.fit(X_train, y_train)
nn.plot_loss()

train_pred = nn.predict(X_train)
test_pred = nn.predict(X_test)

print("Train accuracy is {}".format(nn.acc(y_train, train_pred)))
print("Test accuracy is {}".format(nn.acc(y_test, test_pred)))