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

***
##  Descending into ML