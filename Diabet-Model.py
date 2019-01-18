# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:24:35 2018

@author: Tamer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('diabetes.csv')
dataset = dataset.sort_index()

X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(activation = 'relu', input_dim = 8, units = 4, kernel_initializer = 'uniform'))
classifier.add(Dropout(0.1))

classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))
classifier.add(Dropout(0.1))

classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# New Predictions

X_new = X[3:6, :]
X_new = sc.transform(X_new)
y_new = classifier.predict_classes(X_new)

for i in range(len(X_new)):
    print(X_new[i], " Predicted ", y_new[i])
    
y_new2 = classifier.predict_proba(X_new)

for i in range(len(X_new)):
    print(X_new[i], " Predicted ", y_new2[i])
    
y_new3 = classifier.predict(X_new)

for i in range(len(X_new)):
    print(X_new[i], " Predicted ", y_new3[i])
    
y_new3 = (y_new3 > .5)

# Save and Load Your Keras Deep Learning Models
# https://machinelearningmastery.com/save-load-keras-deep-learning-models/

# Predicting the test set result

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
# Confusion matrix is a summary of prediction results on a classification problem
# Accuracy = correct predictions / total predictions * 100
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# Evaluating the ANN
def build_function():
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 8, units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_function, batch_size = 10, epochs = 100)
accuracy = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv= 10, n_jobs = 1)

mean = accuracy.mean() # 0.7442094135243448
variance = accuracy.std() # 0.04474760490264673

# Tunning the model using grid search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_function(optimizer):
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 8, units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_function)

parameters = {'batch_size' : [10, 25],
              'epochs' : [100, 200],
              'optimizer' : ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# After tunning
def build_function():
    classifier = Sequential()
    classifier.add(Dense(activation = 'relu', input_dim = 8, units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(activation = 'relu', units = 4, kernel_initializer = 'uniform'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_function, batch_size = 10, epochs = 200)
accuracy2 = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv= 10, n_jobs = 1)

mean2 = accuracy2.mean()    # 0.7621893172475914
variance2 = accuracy2.std()  # 0.02097232351757704













