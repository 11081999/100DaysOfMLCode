#https://www.youtube.com/watch?v=vsWrXfO3wWw

#! OBJECTIVE: This program aims to study the concepts of Gradient descent

#! Additionally, we´ll get the m and b values.

#by Roberto Rivera Terán

#Step 0: Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Step 1: Make the Gradient Descent algorithm
def gradient_descent(x, y):
    m_curr, b_curr = 0 

#Step 2: Import dataset.
dataset = pd.read_csv(r'C:\Users\Roberto\Desktop\100DaysOfMLCode\datasets\studentscores.csv')

#!This time around we´ll use numpy arrays
# X: data
X = np.array(dataset.iloc[ : ,   : 1 ].values)
# Y: Target (single dim array)
Y = np.array(dataset.iloc[ : , 1 ].values)

print("\n____________________________________")
print("\n >> Data: \n")
print("X : \n", X)
print("Y: \n", Y)

#Step 2: 

#?  In order to do the "Hot and cold" iterative loss approach-
#?  We would have needed to use the formula y'= b + w1*x1 ...
#?  so that we can have some sort of randomized guesses- 
#?  that would end up minimizing the loss.
#?  we could maybe use an algorithm (https://www.geeksforgeeks.org/linear-regression-python-implementation/)
#?  so that we can pick random values for b and w1 



#?: Done !