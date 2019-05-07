# epiclass
Demonstration of classification on the Epileptic Seizure Recognition Data Set

This shows some exploratory data analysis, followed by training and deploying a few classifiers for the Epileptic Seizure Recognition Data Set.

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

The goal here is to build two models, one to predict seizure vs. non-seizure, and one to predict which one of the five classes a given data point belongs to. So this is a classic supervised classification problem.

## Exploratory data analysis

Let's start by exploring the data. There are no missing values. All the features and the target are integers. 

The data is balanced across all five classes, meaning there are 2300 data points in each class.

Next let's try visualizing the data.

The first thing I noticed is that the features are remarkably homogeneous - they are all similarly scaled.
You can see from this plot that they all have quite similar minima, maxima, means, medians, and interquartile ranges.

![Plot of  minima, maxima, means, medians, and interquartile ranges.](/outputs/feature_summary.png "Ranges and summary statistics of features")

One notable characteristic shown in the above plot is that the maximum value across all the features, 2047, is repeated a significant number of times in the data and is the maximum across a large number of features. Most likely, whatever the measure of brain activity is capped at this value, such as by saturating the measurement device. 2047 is one less than a power of 2 which further substantiates that it's a likely cap.

Let's look a little closer at the interquartile ranges and the mean and median.

![Plot of  means, medians, and interquartile ranges.](/outputs/interquartile.png "Interquartile ranges")

Again all the features are very similar in their scaling, with very similar interquartile ranges. But there are significant outliers in all the features, with values as low as -1885 and as high as 2047. The means and medians are quite close together, so the distributions across each feature are not very skewed.

Let's see if there's any structure to the data. Here's a heatmap of the correlation matrix of the features:

![Correlation heatmap](/outputs/corr_heatmap.png "Correlation heatmap")

Note the diagonal band of high correlation in the centre, flanked by parallel lines of negative correlation, alternating with positive correlation, etc, getting less defined as it moves away from the main diagonal. Clearly the order of the features matters. I am not a neuroscientist, but I theorize that features close to each other in the data set are physically close to each other in the brain, and that this pattern of alternating high and low correlation is characteristic of brain activity. Taking a cut through the correlation matrix at the feature in the centre, X90, you can see the correlation pattern in more detail:

![Correlations with X90](/outputs/corr_X90.png "Correlations with X90")

This shape, with the closest 10 or so features on each side having positive correlation, decreasing as you get farther from the feature, followed by negative correlations with the next 10 or so features, and oscillating afterwards between positive and negative correlations, is characteristic of the correlations with any one feature. This structure should make dimensionality reduction methods, such as Principal Component Analysis (PCA), work quite well on this data. We will revisit this below when we perform PCA on the data.

Before using complex dimensionality reduction methods, I wanted to see if I could see patterns in the classification of the data with some simple feature engineering. With all the features scaled the same, I wanted to see if I could do some new features which were simple statistical summary values on the complete feature set. So I took the mean, median, minimum, maximum, and range (maximum minus minimum) of each data point, summarizing across all the features. There are some pretty clear patterns that should make it not too difficult to write a classifier for this data set. Here, for example, is a box plot of the range across the target classes:

![Box plot of range](/outputs/range_box.png "Boxplot of range of features")

Looking at this boxplot, in the case of the binary classifier (seizure vs all other classes) it seems one could write a naive classifier based only on the range, which would get better-than-guessing performance. The cutoff would be at a range of approx 500 - all data points with higher ranges would be called seizures, and all data points with lower ranges would be other classes. Quite a few of the outliers in the "tumor area" class would likely erroneously be classified as seizures, and some seizures would be classified as non-seizures.

Here's another visualization of the same phenomenon:

![Minimum vs maximum of features by class](/outputs/min_vs_max.png "Minimum vs maximum of features by class")

Clearly seizure activity, and to a lesser extend brain tumor areas, have significantly wider ranges of feature values than normal activity, which tend to stay around the -250 to 250 range.

## Principal Component Analysis

As mentioned when discussing the correlation heatmap, the data seems to have some clear structure. And at 178 features, it could definitely benefit from some dimensionality reduction. So that's why I started with Principal Components Analysis (PCA). First I scaled the data to be between -1 and 1 by dividing it by 2047 - I didn't see any reason to use more sophisticated scaling methods when the parameters were all so similarly scaled. Then I used scikit-learn's PCA class to generate the first 60 PCA components. (60 was an arbitrary choice). First let's take a look at the explained variance ratio. Here I am looking to see how fast the explained variance decays with eigenvalue number and looking for any eigengaps. Here's a semilogy plot of the explained variance across the first 60 PCA components.

![Plot of explained variance ratio of PCA components.](/outputs/var_ratio.png "Explained variance ratio of PCA components")

That's a pretty steady decrease with no clear eigengap, though the decrease in explained variance does get steeper at around 35 PCA components. The other plot I like to see is a semilogy plot of one minus the cumulative sum of the explained variance ratio, which gives you a good idea of how many components you need to include to get various percentages of explained variance.

![Plot of one minus cumulative explained variance ratio of PCA components.](/outputs/var_ratio_sum.png "One minus cumulative explained variance ratio of PCA components")

This shows that you need 32 components to capture more than 90% of the explained variance and 52 components to capture more than 99% of the explained variance. This is not as few components as I would have liked to see, considering the high degree of structure in the data, but it's better than working with the original 178 features.

Here's a plot of the first three PCA components vs the original features:

![Plot of first three PA components](/outputs/pca_components.png "First three PA components")

I then plotted the data points projected onto the first two PCA components, to give an idea of how well the PCA has done in generating a space in which we can separate the classes:
![data points projected onto the first two PCA components](/outputs/two_pca_components.png "data points projected onto the first two PCA components")

It doesn't look linearly separable in only these two dimensions, but it does look like we might be able to use a non-linear method like a support vector machine with a radial basis function kernel or similar to separate the classes.

## Evaluation metrics

We need to decide what metric to use to evaluate our classification algorithms and pipelines.

I am using 70% of the data for training and for hyper-parameter tuning (through cross-validation), and the remaining 30% for testing. The metrics I discuss below are all applied on the test set for evaluating the models.

It's not entirely clear to me what these predictions would be used for, and different applications might require different metrics. For example, if it were being used to more closely monitor patients at risk of epileptic seizure, we would want to more heavily penalize false negatives (classifying a seizure as a non-seizure) than false positives. However, if the classification were used to, for example, automatically apply a treatment that might have side effects, we might want to penalize false positives more than false negatives. For now, without information on how the classifier is to be used, I assume that false negatives and false positives are equally undesirable.

### Accuracy

Classification accuracy is quite appropriate in this case – we are trying to classify as seizure/not seizure or by class, and we want to classify as many correctly as possible. In the multiclass case, the classes are perfectly balanced, meaning there are the same number of each class. So there is no reason to come up with a more complex metric than accuracy. For the seizure vs non-seizure case, the classes are unbalanced, with 20% in one class and 80% in the other. While not extremely unbalanced, accuracy might still not be the best measure, since we could get an accuracy of 80% by naively classifying every data point as a non-seizure. So we may want to consider other metrics.

### Area under the Receiver Operating Characteristic curve

We might want to be able to fine-tune the sensitivity vs specificity of our problem. So for the two-class problem, we might want to consider using the area under the ROC curve

### F1-score

In the two parameter case, the classes are unbalanced, so accuracy may not be an entirely fair metric. Instead we can use the geometric mean of precision and recall, the F1 score, defined as:

2 * precision * recall /(precision + recall)

### Decision

In the end I decided to use the F1-score for the 2-class problem, and accuracy for the 5-class problem. I also record other metrics such as the area under the ROC curve, precision, and recall, but those are not used to choose the model or the hyperparameters.

## Seizure vs not-seizure classification using Principal Components Analysis and Support Vector Machines

Let's start with the seizure (class 1) vs non-seizure (all other classes) classification.

As I mentioned above, after doing PCA, the classes look like they may be separable but not linearly. So the first thing I decided to try was to use a support vector machine (SVM) with a radial-basis-function (RBF) kernel.

I created a pipeline with the PCA and the SVC (support vector classification) using an RBF kernel. The most important hyperparameters are the number of PCA components to use, the penalty parameter C on the SVM error term, and gamma, which is a scale parameter on the RBF function used in the SVM. I used the [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) method with five-fold cross-validation on the training set, optimizing for F1-score, to choose these hyperparameters.

Using 50 PCA components, C=1000, and gamma=0.001 resulted in a cross-validation f1 of 0.961, and an accuracy of 99.5%.

Running that model on the test set (not used in tuning the hyperparameters) results in the following confusion matrix:

| |      Predicted |     |      |
|-----------|------|-----|------|
| **Actual**    | **Not seizure**    | **Seizure**   | **All**  |
| **Not seizure**        | 2726 | 33  | 2759 |
| **Seizure**         | 50   | 641 | 691  |
| **All**       | 2776 | 674 | 3450 |

This comes out to a precision of 98.8%, recall of 98.2%, f1-score of 0.985, and accuracy of 97.6%. This seems like a reasonably good classifier for this problem.

## Multiclass Classification

### Principal Components Analysis and Support Vector Machine

Since we got pretty good results using a PCA and SVM pipeline on the binary classification problem, I also tried the same pipeline for the multiclass classification problem. Again I ran a grid search with 5-fold cross-validation to tune the hyperparameters. Since these classes were balanced, as mentioned above, I used accuracy for the metric when tuning. The optimal hyperparameters came out to be 50 PCA components, C=100 and gamma=0.1

Unfortunately, in this case, I wasn't able to get the accuracy as high as I was able to do in the binary classification problem. The average cross-validation accuracy was 69.8%. On the held-out test set, the accuracy was a similar 71.2%.

Let's look in more detail about why the classifier was not able to get better accuracy than that. Here's the confusion matrix, as a heatmap to show where the classification is failing:

![Confusion matrix for multiclass with PCA and SVM](/outputs/confusion_5c_scaled.png "Confusion matrix for multiclass with PCA and SVMs")

As you can see, the algorithm does pretty well at distinguishing class 1 (seizures) from the rest, like in the binary classification. It has errors of almost every combination, except that it doesn't ever identify a seizure data point as a healthy, eyes-open point. It has trouble telling classes 4 and 5 - eyes open and closed - and especially classes 2 and 3 - measurements in the tumor area and in the healthy part of the brain where there's a tumor elsewhere - apart.

### Random Forest

Since the accuracy was only 70%, even with the best hyperparameters, I wanted to try a different class of method. So I chose a random decision forest. I also ran 5-fold cross-validation on the training set to choose the hyperparameters. Unfortunately, it wasn't an improvement on the PCA and SVM pipeline. The grid search chose a maximum tree depth of 25, and maximum number of features used per tree of 30, and 250 trees. All of these were the maximum number over my grid search and it took 43 seconds to train this forest. The best cross-validation accuracy on the training set was 69.3%. On the held-out test set, the accuracy 68.7%. Here's the confusion matrix on the test set:

![Confusion matrix for random decision forest](/outputs/confusion_5c_rf_scaled.png "Confusion matrix for multiclass with random decision forest")

Similarly to the SVM model above, this model is good at distinguishing seizures from non-seizures but struggles with distinguishing class 2 from class 3 and class 4 from class 5. But it additionally erroneously classifies many class 2, 3, and 4 data points as class 5 and class 5 as class 3.

### Neural network

Since neither the SVM model nor the random forest got the kinds of accuracy I was hoping for, I wanted to see if a neural network could do a better job.

Using keras, I built a dense neural network. I manually tried multiple architectures, training all of them on the training set and measuring their accuracy on the test set. The best architecture I found was a dense converging network with two hidden layers. Here's the architecture, using keras's plot_model method:

![Neural network architecture](/outputs/nn_model.png "Neural network architecture")

Here's the confusion matrix from the resulting neural network:

![Confusion matrix for neural network](/outputs/confusion_nn.png "Confusion matrix for neural network")

Similarly to the SVM model, the neural network does an excellent job of separating out seizure data from the other data points, but struggles to separate classes 2 & 3, and classes 4 & 5. The overall accuracy is 67.7%, similar to but worse than the SVM model and the random forest. 

### Conclusion for multiclass classification

I tried three quite different methods on the multiclass classification problem, spending some significant time tuning each of them. All three, with their best hyperparameters, got accuracies near 70%, with similar patterns in their confusion matrices. I conclude that with the current data set, it may not be possible to create a better multiclass classifier. Classes 2 & 3 have very similar features for this data set, as do classes 4 & 5. I would suggest to the researchers to take more measurements, or summarize their time series differently.

## Deployment and testing

The models I created are saved in the *models* directory for future use. The SVM and random forest models are saved using the [joblib](https://joblib.readthedocs.io/en/latest/) library. If you want to use them, you can load them using joblib.loads, passing the path to the model file you wish to load. The neural network model is saved in keras format and can be loaded using keras.load, again passing the path. Note that the features should be divided by 2047 before passing them to any of the models.

The binary classifier [two_class_pca_svm.z](/models/two_class_pca_svm.z) returns 0 if it's predicted as not a seizure (classes 2, 3, 4, or 5) and 1 if it is predicted as a seizure. The multiclass SVM [five_class_pca_svm.z](/models/five_class_pca_svm.z) and the decision tree [5c_rf_scaled.z](/models/5c_rf_scaled.z) return the predicted class as an integer. The neural network [5c_nn.h5](/models/5c_nn.h5) returns a [one-hot encoding](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/) as used by Keras.

In order to demonstrate how one might deploy the model, I wrote a [web GUI](web_gui.py) to apply the binary PCA-SVM . It creates a form where you can enter features, and then tells you what the classification is. This one does not expect you to divide by 2047 first. If you submit the features into the form, it tells you whether the model predicts a seizure or not. This could also easily be adapted into e.g. a REST API for use by other developers.

I also wrote a [few tests to ensure the models are working](/test_deployment.py) They only test the PCA-SVM models since these were the best models I developed. For each I test a single prediction, and then test that the confusion matrix has not changed. 

## Working with epiclass

### Installing

Epiclass requires python 3. To get epiclass, make sure your python is python 3, and run 

`git clone https://github.com/moink/epiclass/`

`cd epiclass`

`python setup.py install`

### Running

There are three scripts to run, run_epiclass.py, api.py, and test_deployment.py

#### run_epiclass.py

This script is included to show how I created the plots in this readme, trained the models, and tuned the hyperparameters. Running it with all options may take a long time, depending on your hardware -- on the order of hours.

To run it, provide one or more of the following actions as a parameter to the python script:

* explore - make several plots of the data, including of the
                          Principal Component Analysis (PCA) transformation of
                          the features
* pca_svm2 - train a binary classifier (seizure vs
                          non-seizure) using a pipeline of PCA and a support 
                          vector machine
* pca_svm5 - train a multiclass classifier using a pipeline 
                          of PCA and a support vector machine
* rf - train a multiclass classifier using a random decision 
                           forest
* nn - train a multiclass classifier using an artificial 
                           neural network
                  
All the training methods save the models to the model 
directory and additionally save a confusion matrix to the 
                  outputs directory.

For example, to create the data exploration plots and train the neural network, run

`python run_epiclass.py explore nn`

#### test_deployment.py

This runs four tests to ensure that the PCA-SVM models are giving expected results. To run it run:

`python test_deployment.py`

It will show some text in the console indicating whether the tests ran.

#### web_gui.py

This starts a process serving a Web GUI with access to the binary PCA-SVM model.
To run it run:

`python web_gui.py`

Your python will give you a URL - visit it with your browser.

## Conclusion

I think I could improve the models with further tuning (for example, a finer grid search) and by trying other models (e.g. different neural network architectures, logistic regression, fourier transforms). However I think the improvements would be marginal and that I have gotten pretty close to the limits of what the data can provide.
