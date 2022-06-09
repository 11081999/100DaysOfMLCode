#Resources: https://medium.com/machine-learning-with-python/multiple-linear-regression-implementation-in-python-2de9b303fc0c
#Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
#Kaggle: https://www.kaggle.com/code/shivamnijhawan96/wine-quality-using-linear-regression/notebook

#! OBJECTIVE: 
#! Use the same code and methods of Day_3.py but with a different dataset.
#! I believe that the other code didnt work as expected because the data was
#! Not fitted for a linear regression (Actually it was because it was mutivariable).
#by Roberto Rivera Terán

#Step 1: Data Preprocessing
##Step 1.1: Importing the libraries
from tokenize import Double
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Step 1.2: Importing the dataset
dataset = pd.read_csv(r"C:\Users\Roberto\Desktop\100DaysOfMLCode\datasets\winequality-red.csv")
#print("\n____________________________________")
#print("\n >> All data \n")
#print("Data:         \n",    dataset)

#Quality vs Alcohol
#Quality is the last col (the 12th element) on the Y axis
Y= []
#Alcohol is the the 11th element, on the X axis
X= []

col = dataset.iloc[ : , 0].values
for row in col:    
    #print("Row:           \n",    row.split(";"))
    Y.append(float(row.split(";")[12-1]))

    #arr= []
    #for val in row.split(";")[0:-1]:
        #arr.append(float(val))

    X.append(float(row.split(";")[11-1])
)
#! I think all my problems relied on some multivaribalility 
X = np.array(X).reshape(-1, 1)

print("\n____________________________________")
print("\n >> Raw values \n")
# just comparing the last element with the csv will do
print("X (Alcohol):  \n",    X[len(X)-1],        "\n Len:",len(X))
print("Y (Quality):  \n",    Y[len(Y)-1],        "\n Len:",len(Y))

##Step 1.2: Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state = 0)

#X_train needs to be 2D apparently
#! fit and predict need 2d arrays
#X_train= np.array(X_train).reshape(-1, 1)
#X_test = np.array(X_train).reshape(-1, 1)

print("\n____________________________________")
print("\n >> Splitting the dataset into the Training set and Test set \n")

print("X_train Len:",len(X_train))
print("X_test Len:",len(X_test))
print("Y_train Len:",len(Y_train))
print("Y_test: Len:",len(Y_test))

#print("X_train:   \n",    X_train,  "\n Len:",len(X_train))
#print("X_test:    \n",    X_test,   "\n Len:",len(X_test))
#print("Y_train:   \n",    Y_train,  "\n Len:",len(Y_train))
#print("Y_test:    \n",    Y_test,   "\n Len:",len(Y_test))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

##Step 3: Predicting the Test set results
y_pred = regressor.predict(X_test)

print("\n____________________________________")
print("\n >> Prediction \n")
print("Y_pred Len:",len(y_pred))
#print("Y_pred:    \n",    y_pred,   "\n Len:",len(y_pred))

print("\n____________________________________")
print("\n >> Accuracy of our model \n")
print("acc: ",    str(int(round(regressor.score(X_test, Y_test)*100)))+"%", "\n")

#! Note: sometimes X_train is 2d and that wont work on scatter plot
plt.scatter(X,          Y,          color   = 'pink')
plt.scatter(X_test ,    Y_test,     color   = 'green')
plt.scatter(X_train,    Y_train,    color   = 'blue')
plt.plot(   X_test ,    y_pred,     color   = 'red')
plt.xlabel('Alcohol')
plt.ylabel('Quality')
#plt.show()

#? Done!

