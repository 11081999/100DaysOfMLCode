# Google machine learning crash course

***
##  Introduction to Machine Learning

- use statistics and not logic to solve problems.

***
##  Framing

This module investigates how to frame a task as a machine learning problem, and covers many of the basic vocabulary terms shared across a wide range of machine learning (ML) methods.

**(SUPERVISED) Machine Learning** : Supervised learning uses a training set to teach models to yield the desired output. This training dataset includes inputs and correct outputs, which allow the model to learn over time. The algorithm measures its accuracy through the loss function, adjusting until the error has been sufficiently minimized.

Supervised learning can be separated into two types of problems when data **mining—classification** and **regression**...

- ***ML systems learn how to combine input to produce useful predictions on never-before-seen data. ***

[Source](https://www.ibm.com/cloud/learn/supervised-learning)

##  Framing: Key ML Terminology

### Labels
A label is the thing we're predicting—the y variable in simple linear regression. The label could be the future price of wheat, the kind of animal shown in a picture, the meaning of an audio clip, or just about anything.

### Features
A feature is an input variable, the x variable in simple linear regression. A simple machine learning project might use a single feature, while a more sophisticated machine learning project could use millions of features, specified as:

In the spam detector example, the features could include the following:

- words in the email text
- sender's address
- time of day the email was sent
- email contains the phrase "one weird trick."

### Models
A model defines the relationship between features and label. For example, a spam detection model might associate certain features strongly with "spam". Let's highlight two phases of a model's life:

- ***Training*** means creating or learning the model. That is, you show the model labeled examples and enable the model to gradually learn the relationships between features and label.

- ***Inference*** means applying the trained model to unlabeled examples. That is, you use the trained model to make useful predictions (y'). For example, during inference, you can predict *medianHouseValue* for new unlabeled examples.

### Regression vs. classification
A **regression** model predicts continuous values. For example, regression models make predictions that answer questions like the following:

- What is the value of a house in California?
- What is the probability that a user will click on this ad?

A **classification** model predicts discrete values. For example, classification models make predictions that answer questions like the following:

- Is a given email message spam or not spam?
- Is this an image of a dog, a cat, or a hamster?

[Source](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology)
***
##  Descending into ML

### Linear Regression

True, the line on linear regression doesn't pass through every dot, but the line does clearly show the relationship between chirps and temperature. Using the equation for a line, you could write down this relationship as follow:

```
    y= mx + b
```

where:

- ***y*** is the temperature in Celsius—the value we're trying to predict.
- ***m*** is the slope of the line.
- ***x*** is the number of chirps per minute—the value of our input feature.
- ***b*** is the y-intercept.

By convention in machine learning, you'll write the equation for a model slightly differently:

```
    y'= b + w1 * x1
```

where:

- ***y'*** is the predicted label (a desired output).
- ***b*** is the bias (the y-intercept), sometimes referred to as
- ***w1*** is the weight of feature 1. Weight is the same concept as the "slope" ***m*** in the traditional equation of a line.
- ***x1*** is a feature (a known input).

To infer (predict) the temperature ***y'*** for a new chirps-per-minute value ***x1***, just substitute the ***x1*** value into this model.

Although this model uses only one feature, a more sophisticated model might rely on multiple features, each having a separate weight (
***w1***, ***w2***, etc.). For example, a model that relies on three features might look as follows:

```
    y'= b + w1*x1 +  w2*x2 +  w3*x3
```
[Source](https://developers.google.com/machine-learning/crash-course/descending-into-ml/linear-regression)

### Training and Loss

**Training** a model simply means learning (determining) good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

Loss is the penalty for a bad prediction. That is, **loss** is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples.

### Training and Loss | Squared loss

The linear regression models we'll examine here use a loss function called **squared loss** (also known as **L2 loss**). The squared loss for a single example is as follows:

```
    = the square of the difference between the label and the prediction
    = (observation - prediction(x))^2
    = (y - y')^2
```

**Mean square error (MSE)** is the average squared loss per example over the whole dataset. To calculate MSE, sum up all the squared losses for individual examples and then divide by the number of examples:

where:

- **(x, y)** is an example in which

- - **x** is the set of features (for example, chirps/minute, age, gender) that the model uses to make predictions.
- - **y** is the example's label (for example, temperature).

- **prediction(x)** is a function of the weights and bias in combination with the set of features.
- **D** is a data set containing many labeled examples, which are **(x, y)** pairs.
- **N** is the number of examples in **D**.

Although MSE is commonly-used in machine learning, it is neither the only practical loss function nor the best loss function for all circumstances.

[Source](https://developers.google.com/machine-learning/crash-course/descending-into-ml/training-and-loss)
***