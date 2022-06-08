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
    #? remember the formula y' = m*x + b
    m_curr = b_curr = 0
    iterarions = 1000
    n= len(x)

    #Trial and error
    learing_rate= 0.08

    iter= 0
    for i in range(iterarions):

        Y_pred = m_curr*x + b_curr

        cost= (1/n) * sum([val**2 for val in (y-Y_pred)])

        # m¨s derivative 
        md= -(2 / n) * sum(x * (y - Y_pred))
        # b¨s derivative 
        bd= -(2 / n) * sum(y - Y_pred)

        # Apply slight variation to the m and b values.
        m_curr = m_curr - learing_rate * md
        b_curr = b_curr - learing_rate * bd

        #Visulize the adjustment
        plt.plot(x , Y_pred, color = 'red')

        iter= i

    print("\n____________________________________")
    print("\n >> Gradient Descent: \n")
    print("m: {} \nb: {} \nCost: {} \niteration: {} \n".format(
        m_curr, b_curr, cost, iter))

    #?  The graph seems kinds wrong though 
    #?  I believe there should not be any more values above 13
    #?  Since the furthest value of X is 5 

    #Visualize the final adjustment (the most accurate one for predictions)
    plt.plot(x , Y_pred, color = 'blue')
    plt.show()


#Step 2: Import dataset.
dataset = pd.read_csv(r'C:\Users\Roberto\Desktop\100DaysOfMLCode\datasets\studentscores.csv')

#!This time around we´ll use numpy arrays
# X: data
X = np.array(dataset.iloc[ : ,   : 1 ].values)
# Y: Target (single dim array)
Y = np.array(dataset.iloc[ : , 1 ].values)

#Alternative data, Expected m= 2, b= 3
# X: data
X = np.array([1,2,3,4,5])
# Y: Target (single dim array)
Y = np.array([5,7,9,11,13])

print("\n____________________________________")
print("\n >> Data: \n")
print("X : \n", X)
print("Y: \n", Y)

gradient_descent(X, Y)

#?: Done !