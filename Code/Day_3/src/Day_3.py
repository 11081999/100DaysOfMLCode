#
#! OBJECTIVE: 
#! This program aims to create and explain multiple regressions.
#by Roberto Rivera Terán

##Step 1: Data Preprocessing

#? We already did this at Day_1

#Step 1.1: Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Step 1.2: Importing the dataset
dataset = pd.read_csv(r"C:\Users\Roberto\Desktop\100DaysOfMLCode\datasets\50_Startups.csv")
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : ,  4 ].values


#Step 1.3: Encoding (labels) Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

#Step 1.4: Avoiding Dummy Variable Trap
X = X[: , 1:]

#Step 1.5: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Step 2: Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Step 3: Importing the libraries
y_pred = regressor.predict(X_test)

print("\n____________________________________")
print("\n >> Prediction \n")
print("Y_pred:    \n", y_pred)

#!This seems a little sus, needs to be checked
#! x and y must be the same size, idk what´s with that
#plt.scatter(X_test , Y_test, color = 'blue')
#plt.scatter(X_train , Y_train, color = 'red')
#plt.scatter(X, y_pred, color = 'pink')
#plt.show()

#? Done!