#Step 0: Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Step 1: Data Preprocessing

#?  The very same thing we did at "Day_1".

#!  Notice how this time every value is numeric so there is no need for Encoding categorical data.
#!  Nor is there a need to fill the missing values.

dataset = pd.read_csv(r'C:\Users\Roberto\Desktop\100DaysOfMLCode\datasets\studentscores.csv')
# X: data
X = dataset.iloc[ : ,   : 1 ].values
# Y: Target (single dim array)
Y = dataset.iloc[ : , 1 ].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( 
    X, 
    Y, 
    test_size = 1/4, 
    random_state = 0)

print("\n____________________________________")
print("\n >> Split data into training data and validation data \n")
print("X_train:   \n",    X_train,  "\n Len:",len(X_train))
print("X_test:    \n",    X_test,   "\n Len:",len(X_test))
print("Y_train:   \n",    Y_train,  "\n Len:",len(Y_train))
print("Y_test:    \n",    Y_test,   "\n Len:",len(Y_test))

print("X_train Len:",   len(X_train))
print("X_test Len:",    len(X_test))
print("Y_train Len:",   len(Y_train))
print("Y_test: Len:",   len(Y_test))

#Step 2: Fitting Simple Linear Regression Model to the training set

#?  Linear regression is an algorithm used to-
#?  predict or visualize a relationship between two different features/variables.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#Step 3: Predicting the Results
Y_pred = regressor.predict(X_test)

#Step 4: Visualization
##Step 4.2:  Visualizing the results.
plt.scatter(X,          Y,          color   = 'pink')
plt.scatter(X_test ,    Y_test,     color   = 'green')
plt.scatter(X_train,    Y_train,    color   = 'blue')
plt.plot(   X_test ,    Y_pred,     color   = 'red')
#plt.show()

print("\n____________________________________")
print("\n >> Prediction \n")
print("Y_pred:    \n",    Y_pred,   "\n Len:",len(Y_pred))

#?: Done !

#Extra Steps 1: Calculate R^2

#?  R^2 tells us how much of the variation in "Y_AXYS"-
#?  can be explained by "X_AXYS"

from sklearn.metrics import r2_score
R2_test= r2_score(Y_test, Y_pred)
R2_train= r2_score(Y_train, regressor.predict(X_train))

print("\n____________________________________")
print("\n >> How much of the variation in Y can be explained by X \n")
print("R^2: test/train", (R2_test*100), (R2_train*100))
