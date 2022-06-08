
#This program aims to study the concepts of reducing loss
#by Roberto Rivera Terán

#Step 0: Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Regression_Day2 import Regression

#Step 1: Setting an example of Linear Regression

#?  The variable "Y_pred_train" holds the predicted values (y´) 
#?  via linear regression.

reg= Regression()
print("\n____________________________________")
print("\n >> Linear Regression: \n")
print("True values: \n", reg.Y_true)
print("Predicted values: \n", reg.Y_pred)
print("R^2: \n", (reg.R2_train*100))

#Step 2: 

#?  In order to do the "Hot and cold" iterative loss approach-
#?  We would have needed to use the formula y'= b + w1*x1 ...
#?  so that we can have some sort of randomized guesses- 
#?  that would end up minimizing the loss.
#?  we could maybe use an algorithm (https://www.geeksforgeeks.org/linear-regression-python-implementation/)
#?  so that we can pick random values for b and w1 

#Step 3: ACTUALLY compute the loss

#?  In this example the loss function well use will be the.
#?  The  loss is a number indicating how bad the model's prediction-
#?  was on a single example. If the model's prediction is perfect- 
#?  the loss is zero; otherwise, the loss is greater. 
#?  The goal of training a model is to find a set of weights and biases-
#?  that have low loss, on average, across all examples.

from sklearn.metrics import mean_squared_error
#! I did not use the Y_pred_train because it gave an error XD
loss= mean_squared_error(reg.Y_true, reg.Y_pred)
print("\n____________________________________")
print("\n >>  Mean squared error regression loss. \n")
print("Loss: \n", loss)

#? The loss is pretty high for an R^2 so reliable, the next step
#? Would be to adjust the prediction formula witch I cant because
#? I used sklearn library. 

#?: Done !