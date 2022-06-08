#Step 0: Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#! This class is the code made for the regresion example at "Day_2".
class Regression:

    def __init__(self):
        #Step 1: Data Preprocessing
        #?  The very same thing we did at "Day_1".
        #!  Notice how this time every value is numeric so there is no need for Encoding categorical data.
        #!  Nor is there a need to fill the missing values.
        dataset = pd.read_csv(r'C:\Users\Roberto\Desktop\100DaysOfMLCode\datasets\studentscores.csv')
        # X: data
        X = dataset.iloc[ : ,   : 1 ].values
        # Y: Target (single dim array)
        Y = dataset.iloc[ : , 1 ].values

        self.Y_true= Y

        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split( 
            X, 
            Y, 
            test_size = 1/4, 
            random_state = 0)

        self.X_train    = X_train
        self.X_test     = X_test
        self.Y_train    = Y_train
        self.Y_test     = Y_test

        #Step 2: Fitting Simple Linear Regression Model to the training set
        #?  Linear regression is an algorithm used to-
        #?  predict or visualize a relationship between two different features/variables.
        from sklearn.linear_model import LinearRegression
        self.regressor = LinearRegression()
        self.regressor = self.regressor.fit(X_train, Y_train)

        #Step 3: Predicting the Results
        self.Y_pred_test = self.regressor.predict(X_test)
        self.Y_pred_train = self.regressor.predict(X_train)
        self.Y_pred = self.regressor.predict(X)

        #Step 4: Visualization
        #plt.subplot(1, 2, 1)  # row 1, column 2, count 1

        ##Step 4.1:  Visualizing the training results.
        #plt.scatter(X_train , Y_train, color = 'red')
        #plt.plot(X_train , regressor.predict(X_train), color ='blue', )


        ##Step 4.2:  Visualizing the test results.
        #plt.scatter(X_test , Y_test, color = 'green')
        #plt.plot(X_test , Y_pred, color ='orange')

        #plt.subplot(1, 2, 2)    # row 1, column 2, count 2
        #plt.scatter(X , Y, color = 'pink')

        #plt.show()

        #?: Done !

        #Extra Steps 1: Calculate R^2

        #?  R^2 tells us how much of the variation in "Y_AXYS"-
        #?  can be explained by "X_AXYS"

        from sklearn.metrics import r2_score
        self.R2_test= r2_score(Y_test, self.Y_pred_test)
        self.R2_train= r2_score(Y_train, self.Y_pred_train)



    def printResults(self):
        print("\n____________________________________")
        print("\n >> Prediction \n")
        print("(X_train) Y_pred:    \n", self.Y_pred_train)
        print("(X_test) Y_pred:     \n", self.Y_pred_test)

        print("\n____________________________________")
        print("\n >> How much of the variation in Y can be explained by X \n")
        print("R^2: test/train", (self.R2_test*100), (self.R2_train*100))

