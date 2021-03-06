# Data PreProcessing
***
<p align="center">
  <img src="https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Info-graphs/Day%201.jpg">
</p>
# INSTRUCTIONS

As shown in the infograph we will break down data preprocessing in 6 essential steps.
Get the dataset from [here](https://github.com/Avik-Jain/100-Days-Of-ML-Code/tree/master/datasets) that is used in this example

## Step 1: Importing the libraries
```Python
import numpy as np
import pandas as pd
```
## Step 2: Importing dataset
```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
```
## Step 3: Handling the missing data
```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
```
## Step 4: Encoding categorical data
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
```
### Creating a dummy variable
```python
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```
## Step 5: Splitting the datasets into training sets and Test sets 
```python
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```

## Step 6: Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
```
### Done :smile:

***
### UPDATED CODE by me as of 2022

#Step 1: Importing the libraries
```Python
import numpy as np
import pandas as pd
```

#Step 2: Importing dataset
```Python
dataset = pd.read_csv(r"C:\Users\Roberto\Desktop\100DaysOfMLCode\Code\Day_1\src\Data.csv")
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
```

#Step 3: Handling the missing data
```Python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = "mean")
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
```


#Step 4: Encoding categorical data
```Python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
```

#Step 4.1: Creating a dummy variable
```Python
#onehotencoder = OneHotEncoder()
#X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```

#Step 5: Splitting the datasets into training sets and Test sets
```Python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( 
    X , Y , test_size = 0.2, random_state = 0, stratify=Y)
```

#Step 6: Feature Scaling, standarization (can also use normalization)
```Python
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
#Standarized values of X_train & Test_X
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.fit_transform(X_test)
```
#Step 7: Make a graph (?), maybe next time...

#Done !