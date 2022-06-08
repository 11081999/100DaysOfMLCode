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
print("X_train:         \n",    X_train)
print("X_test:          \n",    X_test)
print("Y_train:         \n",    Y_train)
print("Y_test:          \n",    Y_test)

#Step 2: Fitting Simple Linear Regression Model to the training set

#?  Linear regression is an algorithm used to-
#?  predict or visualize a relationship between two different features/variables.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#Step 3: Predicting the Results
Y_pred = regressor.predict(X_test)

#Step 4: Visualization
plt.subplot(1, 2, 1)  # row 1, column 2, count 1

##Step 4.1:  Visualizing the training results.
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue', )


##Step 4.2:  Visualizing the test results.
plt.scatter(X_test , Y_test, color = 'green')
plt.plot(X_test , Y_pred, color ='orange')

plt.subplot(1, 2, 2)    # row 1, column 2, count 2
plt.scatter(X , Y, color = 'pink')

#plt.show()

print("\n____________________________________")
print("\n >> Prediction \n")
print("(X_train) Y_pred:         \n",    regressor.predict(X_train))
print("(X_test) Y_pred:         \n",    Y_pred)

#?: Done !