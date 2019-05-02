# epiclass
Demonstration of classification on the Epileptic Seizure Recognition Data Set

This shows some exploratory data analysis, followed by training and deploying a classifier for the Epileptic Seizure Recognition Data Set.

The data set is available from https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

There are 11500 data points (rows in the data set) and 178 features. There are five classes for the data. Quoting from the above link:

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

The data is complete - there are no missing values.

The data is balanced across all five classes, meaning there are 2300 data points in each class.

## Exploratory data analysis

Let's start by visualizing the data.

The first thing I noticed is that the features are remarkably homogeneous - they are all similarly scaled.
You can see from this plot that they all have quite similar minima, maxima, means, medians, and interquartile ranges.

![Plot of  minima, maxima, means, medians, and interquartile ranges.](/outputs/feature_summary.png "Ranges and summary statistics of features")
