# Google machine learning crash course pt. 2
***
##  Generalization.

Generalization refers to your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.
### Generalization | Peril of Overfitting
An **overfit** model gets a low loss during training but does a poor job predicting new data. If a model fits the current sample well, how can we trust that it will make good predictions on new data? As you'll [see later on](https://developers.google.com/machine-learning/crash-course/regularization-for-simplicity/l2-regularization), overfitting is caused by making a model more complex than necessary. The fundamental tension of machine learning is between fitting our data well, but also fitting the data as simply as possible.

Machine learning's goal is to predict well on new data drawn from a (hidden) true probability distribution. Unfortunately, the model can't see the whole truth; the model can only sample from a training data set. If a model fits the current examples well, how can you trust the model will also make good predictions on never-before-seen examples?
William of Ockham, a 14th century friar and philosopher, loved simplicity. He believed that scientists should prefer simpler formulas or theories over more complex ones. To put Ockham's razor in machine learning terms:

```
The less complex an ML model, the more likely that a good empirical result is not just due to the peculiarities of the sample.
```

In modern times, we've formalized Ockham's razor into the fields of statistical learning theory and computational learning theory. These fields have developed **generalization bounds**--a statistical description of a model's ability to generalize to new data based on factors such as:

- the complexity of the model
- the model's performance on training data

While the theoretical analysis provides formal guarantees under idealized assumptions, they can be difficult to apply in practice. Machine Learning Crash Course focuses instead on empirical evaluation to judge a model's ability to generalize to new data.

A machine learning model aims to make good predictions on new, previously unseen data. But if you are building a model from your data set, how would you get the previously unseen data? Well, one way is to divide your data set into two subsets:

- **training set** -a subset to train a model.
- **test set** -a subset to test the model.

Good performance on the test set is a useful indicator of good performance on the new data in general, assuming that:

- The test set is large enough.
- You don't cheat by using the same test set over and over.

### Generalization | The ML fine print

The following three basic assumptions guide generalization:

- We draw examples **independently and identically** (i.i.d) at random from the distribution. In other words, examples don't influence each other. (An alternate explanation: i.i.d. is a way of referring to the randomness of variables.)
- The distribution is **stationary**; that is the distribution doesn't change within the data set.
- We draw examples from partitions from the **same distribution**.

In practice, we sometimes violate these assumptions. For example:

- Consider a model that chooses ads to display. The i.i.d. assumption would be violated if the model bases its choice of ads, in part, on what ads the user has previously seen.
- Consider a data set that contains retail sales information for a year. User's purchases change seasonally, which would violate stationarity.

When we know that any of the preceding three basic assumptions are violated, we must pay careful attention to metrics.

[Source](https://developers.google.com/machine-learning/crash-course/generalization/peril-of-overfitting)
***
##  Training and Test Sets

### Training and Test Sets | Splitting Data
Make sure that your test set meets the following two conditions:

- Is large enough to yield statistically meaningful results.
- Is representative of the data set as a whole. In other words, don't pick a test set with different characteristics than the training set.

Assuming that your test set meets the preceding two conditions, your goal is to create a model that generalizes well to new data. Our test set serves as a proxy for new data. For example, consider the [following figure](https://developers.google.com/machine-learning/crash-course/images/TrainingDataVsTestData.svg). Notice that the model learned for the training data is very simple. This model doesn't do a perfect job—a few predictions are wrong. However, this model does about as well on the test data as it does on the training data. In other words, this simple model does not overfit the training data.

```diff 
- Never train on test data.
``` 

If you are seeing surprisingly good results on your evaluation metrics, it might be a sign that you are accidentally training on the test set. For example, high accuracy might indicate that test data has leaked into the training set.

For example, consider a model that predicts whether an email is spam, using the subject line, email body, and sender's email address as features. We apportion the data into training and test sets, with an 80-20 split. After training, the model achieves 99% precision on both the training set and the test set. We'd expect a lower precision on the test set, so we take another look at the data and discover that many of the examples in the test set are duplicates of examples in the training set (we neglected to scrub duplicate entries for the same spam email from our input database before splitting the data). We've inadvertently trained on some of our test data, and as a result, we're no longer accurately measuring how well our model generalizes to new data.

[Source](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data)
***
##  Validation Set

### Validation Set | Another Partition
The previous module introduced partitioning a data set into a training set and a test set. This partitioning enabled you to train on one set of examples and then to test the model against a different set of examples. With two partitions, the workflow could look as follows:

<div style="background-color:grey;">
    <p align="center">
        <img src="https://developers.google.com/machine-learning/crash-course/images/WorkflowWithTestSet.svg">
    </p>
</div>

In the figure, "Tweak model" means adjusting anything about the model you can dream up—from changing the learning rate, to adding or removing features, to designing a completely new model from scratch. At the end of this workflow, you pick the model that does best on the test set.

Dividing the data set into two sets is a good idea, but not a panacea. You can greatly reduce your chances of overfitting by partitioning the data set into the three subsets shown in the following figure:

<div style="background-color:grey;">
    <p align="center">
        <img src="https://developers.google.com/machine-learning/crash-course/images/PartitionThreeSets.svg">
    </p>
</div>

Use the validation set to evaluate results from the training set. Then, use the test set to double-check your evaluation after the model has "passed" the validation set. The following figure shows this new workflow:

<div style="background-color:grey;">
    <p align="center">
        <img src="https://developers.google.com/machine-learning/crash-course/images/WorkflowWithValidationSet.svg">
    </p>
</div>

In this improved workflow:

- Pick the model that does best on the validation set.
- Double-check that model against the test set.

This is a better workflow because it creates fewer exposures to the test set.

[Source](https://developers.google.com/machine-learning/crash-course/validation/another-partition)
***
##  Representation

A machine learning model can't directly see, hear, or sense input examples. Instead, you must create a representation of the data to provide the model with a useful vantage point into the data's key qualities. That is, in order to train a model, you must choose the set of features that best represent the data.

### Representation | Feature Engineering

Feature engineering means transforming raw data into a feature vector. Expect to spend significant time doing feature engineering.

Many machine learning models must represent the features as real-numbered vectors since the feature values must be multiplied by the model weights.

Mapping numeric values
Integer and floating-point data don't need a special encoding because they can be multiplied by a numeric weight.

### Mapping categorical values
Categorical features have a discrete set of possible values. For example, there might be a feature called street_name with options that a String.

Since models cannot multiply strings by the learned weights, we use feature engineering to convert strings to numeric values.

We can accomplish this by defining a mapping from the feature values, which we'll refer to as the vocabulary of possible values, to integers. Since not every street in the world will appear in our dataset, we can group all other streets into a catch-all "other" category, known as an OOV (out-of-vocabulary) bucket.

However, if we incorporate these index numbers directly into our model, it will impose some constraints that might be problematic:

- We'll be learning a single weight that applies to all streets. For example, if we learn a weight of 6 for street_name, then we will multiply it by 0 for Charleston Road, by 1 for North Shoreline Boulevard, 2 for Shorebird Way and so on. Consider a model that predicts house prices using street_name as a feature. It is unlikely that there is a linear adjustment of price based on the street name, and furthermore this would assume you have ordered the streets based on their average house price. Our model needs the flexibility of learning different weights for each street that will be added to the price estimated using the other features.

- We aren't accounting for cases where street_name may take multiple values. For example, many houses are located at the corner of two streets, and there's no way to encode that information in the street_name value if it contains a single index.

To remove both these constraints, we can instead create a binary vector for each categorical feature in our model that represents values as follows:

- For values that apply to the example, set corresponding vector elements to 1.
- Set all other elements to 0.

The length of this vector is equal to the number of elements in the vocabulary. This representation is called a one-hot encoding when a single value is 1, and a multi-hot encoding when multiple values are 1.

This approach effectively creates a Boolean variable for every feature value (e.g., street name). Here, if a house is on Shorebird Way then the binary value is 1 only for Shorebird Way. Thus, the model uses only the weight for Shorebird Way.

Similarly, if a house is at the corner of two streets, then two binary values are set to 1, and the model uses both their respective weights. 

### Sparse Representation
Suppose that you had 1,000,000 different street names in your data set that you wanted to include as values for street_name. Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a very inefficient representation in terms of both storage and computation time when processing these vectors. In this situation, a common approach is to use a sparse representation in which only nonzero values are stored. In sparse representations, an independent model weight is still learned for each feature value, as described above.

[Source](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)

### Representation | Qualities of Good Features
We've explored ways to map raw data into suitable feature vectors, but that's only part of the work. We must now explore what kinds of values actually make good features within those feature vectors.

- Avoid rarely used discrete feature values

***Good feature values should appear more than 5 or so times*** in a data set. Doing so enables a model to learn how this feature value relates to the label. That is, having many examples with the same discrete value gives the model a chance to see the feature in different settings, and in turn, determine when it's a good predictor for the label. For example, a house_type feature would likely contain many examples in which its value was victorian:

- house_type: victorian ✔

Conversely, if a feature's value appears only once or very rarely, the model can't make predictions based on that feature. For example, unique_house_id is a bad feature because each value would be used only once, so the model couldn't learn anything from it:

- house_type: victorian ✘

### Prefer clear and obvious meanings
Each feature should have a clear and obvious meaning to anyone on the project. For example, the following good feature is clearly named and the value makes sense with respect to the name:

-  house_age_years: 27 ✔

Conversely, the meaning of the following feature value is pretty much indecipherable to anyone but the engineer who created it:

-  house_age: 851472000 ✘

In some cases, noisy data (rather than bad engineering choices) causes unclear values. For example, the following user_age_years came from a source that didn't check for appropriate values:

-  user_age_years: 277 ✘

### Don't mix "magic" values with actual data
Good floating-point features don't contain peculiar out-of-range discontinuities or "magic" values. For example, suppose a feature holds a floating-point value between 0 and 1. So, values like the following are fine:

-  quality_rating: 0.82 ✔
-  quality_rating: 0.37 ✔

However, if a user didn't enter a quality_rating, perhaps the data set represented its absence with a magic value like the following:

-  quality_rating: -1.0 ✘

To explicitly mark magic values, create a Boolean feature that indicates whether or not a quality_rating was supplied. Give this Boolean feature a name like is_quality_rating_defined.

In the original feature, replace the magic values as follows:

    For variables that take a finite set of values (discrete variables), add a new value to the set and use it to signify that the feature value is missing.
    For continuous variables, ensure missing values do not affect the model by using the mean value of the feature's data.

Account for upstream instability
The definition of a feature shouldn't change over time. For example, the following value is useful because the city name probably won't change. (Note that we'll still need to convert a string like "br/sao_paulo" to a one-hot vector.)

-  city_id: "br/sao_paulo" ✔

But gathering a value inferred by another model carries additional costs. Perhaps the value "219" currently represents Sao Paulo, but that representation could easily change on a future run of the other model:

-  inferred_city_cluster: "219" ✘

[Source](https://developers.google.com/machine-learning/crash-course/representation/qualities-of-good-features)