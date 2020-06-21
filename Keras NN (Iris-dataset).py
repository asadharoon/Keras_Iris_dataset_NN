# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:20:36 2020

@author: Asad
"""

#Neural Network

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder

data_set_url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

columns=['sepal-length','sepal-width','petal-length','petal-width','Class']

dataset=pd.read_csv(data_set_url,names=columns)

X=dataset.iloc[:,:-1]
# Scale data to have mean 0 and variance 1 
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


y_class=dataset.iloc[:,4]


n_features = X.shape[1]
n_classes = 1

from sklearn.preprocessing import LabelEncoder
classes=LabelEncoder()
classes.fit(y_class)
y_new=classes.transform(y_class)
y=y_new


#Train test split
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=1)

import keras
from keras.models import Sequential
from keras.layers import Dense
# Neural network
model = Sequential()

n=4
for i in range(n):
    model.add(Dense(10, input_dim=n_features, activation='relu'))


model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='cosine_similarity', optimizer='adam', metrics=['accuracy'])

# fit in model
history=model.fit(X_train, y_train, epochs=100,batch_size=200,verbose=0)

# predict
y_predictions = model.predict(X_test)


s=model.evaluate(X_test,y_test,verbose=0)
print("Accuracy is ",s[1])
