# epiclass
Demonstration of classification on the Epileptic Seizure Recognition Data Set

This shows some exploratory data analysis, followed by training and deploying a classifier for the Epileptic Seizure Recognition Data Set.

The data set is available from https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

There are 11500 data points (rows in the data set) and 178 features (columns). There are five classes for the data. Quoting from the above link:

> y contains the category of the 178-dimensional input vector. Specifically y in {1, 2, 3, 4, 5}: 
> 
> * 5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open 
> 
> * 4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed 
> 
> * 3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area 
> 
> * 2 - They recorder the EEG from the area where the tumor was located 
> 
> * 1 - Recording of seizure activity 

The goal here is to build two models, one to predict seizure vs. non-seizure, and one to predict which one of the five classes a given datapoint belongs to. So this is a classic classification problem.

## Exploratory data analysis

Let's start by exploring the data. Using the pandas info() and describe() methods to start, it shows that the data is complete - there are no missing values. All the features and the target are integers. 

The data is balanced across all five classes, meaning there are 2300 data points in each class.

Next let's try visualizing the data.

The first thing I noticed is that the features are remarkably homogeneous - they are all similarly scaled.
You can see from this plot that they all have quite similar minima, maxima, means, medians, and interquartile ranges.

![Plot of  minima, maxima, means, medians, and interquartile ranges.](/outputs/feature_summary.png "Ranges and summary statistics of features")

One notable characteristic shown in the above plot is that the maximum value across all the features, 2047, is repeated a significant number of times in the data and is the maximum across a large number of features. Most likely, whatever the measure of brain activity is capped at this value, such as by saturating the measurement device. It's notable that 2047 is one less than a power of 2 which further substantiates that it's a likely cap.

Let's look a little closer at the interquartile ranges and the mean and median.

![Plot of  means, medians, and interquartile ranges.](/outputs/interquartile.png "Interquartile ranges")

Again all the features are very similar in their scaling, with interquartile ranges between -53 or -54 and 36 or 37. But there are significant outliers in all the features, with values as low as -1885 and as high as 2047. The means and medians are quite close together, so the distributions across each feature are not very skewed.

Let's see if there's any structure to the data. Here's a heatmap of the correlation matrix of the features:

![Correlation heatmap](/outputs/corr_heatmap.png "Correlation heatmap")

Note the diagonal band of high correlation in the centre, flanked by parallel lines of negative correlation, alternating with positive correlation, etc, getting less defined as it moves away from the main diagonal. Clearly the order of the features matters. I am not a neuroscientist, but I theorize that features close to each other in the data set are physically close to each other in the brain, and that this pattern of alternating high and low correlation is characteristic of brain activity. Taking a cut through the correlation matrix at the feature in the centre, X90, you can see the correlation pattern in more detail:

![Correlations with X90](/outputs/corr_X90.png "Correlations with X90")

This shape, with the closest 10 or so features on each side having positive correlation, decreasing as you get farther from the feature, followed by negative correlations with the next 10 or so features, and oscillating afterwards between positive and negative correlations, is characteristic of the correlations with any one feature. This structure should make dimensionality reduction methods, such as Principal Component Analysis (PCA), work quite well on this data. We will revisit this below when we perform PCA on the data.

Before using complex dimensionality reduction methods, I wanted to see if I could see patterns in the classification of the data with some simple feature engineering. With all the features scaled the same, I wanted to see if I could do some new features which were simple statistical summary values on the complete feature set. So I took the mean, median, minimum, maximum, and range (maximum minus minimum) of each datapoint, summarizing across all the features. There are some pretty clear patterns that should make it not too difficult to write a classifier for this data set. Here, for example, is a box plot of the range across the target classes:

![Box plot of range](/outputs/range_box.png "Boxplot of range of features")

Looking at this boxplot, in the case of the binary classifier (seizure vs all other classes) it seems one could write a naive classifier based only on the range, which would get better-than-guessing performance. The cutoff would be at a range of approx 500 - all data points with higher ranges would be called seizures, and all data points with lower ranges would be other classes. Quite a few of the outliers in the "tumor area" class would likely erroneously be classified as seizures, and some seizures would be classified as non-seizures.

Here's another visualization of the same phenomenon:

![Minimum vs maximum of features by class](/outputs/min_vs_max.png "Minimum vs maximum of features by class")

Clearly seizure activity, and to a lesser extend brain tumor areas, have significantly wider ranges of feature values than normal activity, which tend to stay around the -250 to 250 range.

## Principal Components Analysis

As mentioned when discussing the correlation heatmap, the data seems to have some clear structure. And at 178 features, it could definitely benefit from some dimensionality reduction. So that's why I started with Principal Components Analysis (PCA). I used scikit-learn's PCA class to generate the first 60 PCA components. (60 was an arbitrary choice). First let's take a look at the explained variance ratio. Here I am looking to see how fast the explained variance decays with eigenvalue number and looking for any eigengaps. Here's a semilogy plot of the explained variance across the first 60 PCA components.

![Plot of explained variance ratio of PCA components.](/outputs/var_ratio.png "Explained variance ratio of PCA components")

That's a pretty steady decrease with no clear eigengap, though the decrease in explained variance does get steeper at around 35 PCA components. The other plot I like to see is a semilogy plot of one minus the cumulative sum of the explained variance ratio, which gives you a good idea of how many components you need to include to get various percentages of explained variance.

![Plot of one minus cumulative explained variance ratio of PCA components.](/outputs/var_ratio_sum.png "One minus cumulative explained variance ratio of PCA components")

This shows that you need 32 components to capture more than 90% of the explained variance and 52 components to capture more than 99% of the explained variance. This is not as few components as I would have liked to see, considering the high degree of structure in the data, but it's better than working with the original 178 features.

Here's a plot of the first five PCA components vs the original features:

![Plot of first five PA components](/outputs/pca_components.png "First five PA components")

You can see the how the data structure that we saw above in the heatmap and the correlations with X90 have affected the PCA components - the components show that same alternating pattern with the same wavelength.

I then plotted the datapoints projected onto the first two PCA components, to give an idea of how well the PCA has done in generating a space in which we can separate the classes:
![Datapoints projected onto the first two PCA components](/outputs/two_pca_components.png "Datapoints projected onto the first two PCA components")

Doesn't look linearly separable in only these two dimensions, but it does look like we might be able to use a non-linear method like a support vector machine with a radial basis function kernel or similar to separate the classes.

## Evaluation metrics

We need to decide what metric to use to evaluate our classification algorithms and pipelines.

I am using 70% of the data for training and for hyper-parameter tuning (through cross-validation), and the remaining 30% for testing. The metrics I discuss below are all applied on the test set for evaluating the models.

### Accuracy

Classification accuracy is quite appropriate in this case – we are trying to classify as seizure/not seizure or by class, and we want to classify as many correctly as possible. In the multiclass case, the classes are perfectly balanced, meaning there are the same number of each class. So there is no reason to come up with a more complex metric than accuracy. For the seizure vs non-seizure case, the classes are unbalanced, with 20% in one class and 80% in the other. While not extremely unbalanced, it’s not balanced either. So we may want to consider other metrics.

## Area under the Receiver Operating Characteristic curve

We might want to be able to fine-tune the sensitivity vs specificity of our problem. So for the two-class problem, we might want to consider using the area under the ROC curve

### F1-score

In the two parameter case, the classes are unbalanced, so accuracy may not be an entirely fair metric. Instead we can use the geometric mean of precision and recall, the F1 score, defined as:

2 * precision * recall /(precision + recall)

### Decision

In the end I decided to use the F1-score for the 2-class problem, and accuracy for the 5-class problem. I also record other metrics such as the area under the ROC curve, precision, and recall, but those are not used to choose the model or the hyperparameters.

## Seizure vs not-seizure classification using Principal Components Analysis and Support Vector Machines

Let's start with the seizure (class 1) vs non-seizure (all other classes) classification.

As I mentioned above, after doing PCA, the classes look like they may be separable but not linearly. So the first thing I decided to try was to use a support vector machine (SVM) with a radial-basis-function (RBF) kernel.

I created a pipeine with the PCA and the SVC (support vector classification) using an RBF kernel. The most important hyperparameters are the number of PCA components to use, the penalty parameter C on the SVM error term, and gamma, which is a scale parameter on the RBF function used in the SVM. I used the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) method with five-fold cross-validation on the training set, optimizing for F1-score, to choose these hyperparameters.

Using 50 PCA components, C = 1000, and gamma = 0.001, the model has a mean crossvalidation accuracy on the test set of 93.75%  and a mean test F1 of 93.83%. In this case, both the accuracy and the F1 metric agree on which set of hyperparameters are optimal.

Running that model on the test set (not used in tuning the hyperparameters) results in the following confusion matrix:

## Multiclass Classification

Since we got pretty good results using a PCA and SVM pipeline on the binary classification problem, I also tried the same pipeline for the multiclass classification problem.
