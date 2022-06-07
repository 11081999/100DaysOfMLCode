#!Step 1: Importing the libraries
import numpy as np
import pandas as pd

#Step 2: Importing dataset
dataset = pd.read_csv(r"C:\Users\Roberto\Desktop\100DaysOfMLCode\Code\Day_1\src\Data.csv")

# X: data
X = dataset.iloc[ : , :-1].values
# Y: Target (single dim array)
Y = dataset.iloc[ : , 3].values

print("DATASET: \n", dataset)
print("\n____________________________________")
print("X:       \n", X)
print("Y:       \n", Y)

#Step 3: Handling the missing data

#?  Replace the missing data by the mean or median of the entire column
#?  This is the use of the Imputer of sklearn().

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = "mean")
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])

print("\n____________________________________")
print("\n >> nan values are replced with the mean \n")
print("Imputer_X:       \n", X)


#Step 4: Encoding categorical data

#?  Categorical data are variables that ocntain label values rather 
#?  than numeric. Values such as "Yes" or "No" cannot be used in 
#?  mathematical models so these values are encoded into numbers.
#?  That is the purpose of the LabelEncoder(). 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])

print("____________________________________")
print("\n >> label values are encoded into numbers \n")
print("LabelEncoder_X:       \n", X)

#Step 4.1: Creating a dummy variable
#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)

#print("\n____________________________________")
#print("\n >> label values are encoded into numbers \n")
#print("HotEncoder_X:       \n", X)
print("HotEncoder_Y:       \n", Y)

#Step 5: Splitting the datasets into training sets and Test sets

#?  Split data into training data and validation data 
#?  using train_test_split()

from sklearn.model_selection import train_test_split
# Note: Only 20% of all the data will be used as testing data
# While the remaining 80% will be training data.
X_train, X_test, Y_train, Y_test = train_test_split( 
    X , Y , test_size = 0.2, random_state = 0, stratify=Y)

print("\n____________________________________")
print("\n >> Split data into training data and validation data \n")
print("X_train:         \n",    X_train)
print("X_test:          \n",    X_test)
print("Y_train:         \n",    Y_train)
print("Y_test:          \n",    Y_test)

#Step 6: Feature Scaling, standarization (can also use normalization)
from sklearn.preprocessing import StandardScaler

#?  scales the features of our data 
#?  so that they all have a similar range.

scaler_X = StandardScaler()
#Standarized values of X_train & Test_X
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_test)

print("\n____________________________________")
print("\n >> Scaled \n")
print("X_train, scaled:     \n", X_train)
print("X_test,  scaled:     \n", X_test)

#Step 7: Make a graph (?), maybe next time...

#?: Done !