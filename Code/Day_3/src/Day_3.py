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
print("\n____________________________________")
print("\n >> All data \n")
print("Data:         \n",    dataset)

# X: data
X = dataset.iloc[ : , :-1].values
# Y: Target (single dim array)
Y = dataset.iloc[ : ,  4 ].values

print("\n____________________________________")
print("\n >> Raw values \n")
print("X:           \n",    X,        "\n Len:",len(X))
print("Y (Profit):  \n",    Y,        "\n Len:",len(Y))

#Step 1.3: Encoding (labels) Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

#Encode the "State" column with a number
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X).toarray()

print("\n____________________________________")
print("\n >> Encoding (labels) Categorical data \n")
print("X:         \n",    X,        "\n Len:",len(X))
print("Y:         \n",    Y,        "\n Len:",len(Y))

#Step 1.4: Avoiding Dummy Variable Trap

#?  The dummy variable trap is a scenario in which two
#?  or more variables are highly correlated; in simple terms,
#?  one variable can be predicted from the others.
#?  Intuitively, there is a duplicate category...

#?  This is, if we drop the male category it is inherently defined in the female
#?  Category (zero female value indicate male, and vice versa)

#? In this example we drop 

X = X[: , 1:]

print("\n____________________________________")
print("\n >> Avoiding Dummy Variable Trap \n")
print("X:         \n",    X,        "\n Len:",len(X))

#Step 1.5: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 0)

print("\n____________________________________")
print("\n >> Splitting the dataset into the Training set and Test set \n")
print("X_train:   \n",    X_train,  "\n Len:",len(X_train))
print("X_test:    \n",    X_test,   "\n Len:",len(X_test))
print("Y_train:   \n",    Y_train,  "\n Len:",len(Y_train))
print("Y_test:    \n",    Y_test,   "\n Len:",len(Y_test))

#Step 2: Fitting Multiple Linear Regression to the Training set

#?  In a nutshell: fitting is equal to training. Then, after it is trained, 
#?  the model can be used to make predictions, usually with a .predict() method call
#?https://stackoverflow.com/questions/45704226/what-does-the-fit-method-in-scikit-learn-do

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#X_train needs to be 2D apparently
regressor.fit(X_train, Y_train)

#Step 3: Predicting the Test set results
y_pred = regressor.predict(X_test)

print("\n____________________________________")
print("\n >> Prediction \n")
print("Y_pred:    \n",    y_pred,   "\n Len:",len(y_pred))

print("\n____________________________________")
print("\n >> Accuracy of our model \n")
print("acc: ",    regressor.score(X_test,Y_test)*100)

#! Note: sometimes X_train is 2d and that wont work on scatter plot
plt.scatter(X_test[:,0],    Y_test, color   = 'blue')
plt.scatter(X_train[:,0],   Y_train, color  = 'red')

#! Missing: the graph seems wrong, maybe the prediction data is not up to a linear regression.
#! Ill advice using the same code on a different more "lineal" dataset
plt.plot(   X_test[:,0],    y_pred, color   = 'black')
#plt.show()

#? Done!