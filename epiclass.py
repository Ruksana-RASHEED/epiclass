"""Visualize seizure data set and create prediction models

Demonstration of classification on the Epileptic Seizure Recognition Data Set

This shows some exploratory data analysis, followed by training and deploying a
classifier for the Epileptic Seizure Recognition Data Set.

The data set is available from
https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition

There are 11500 data points (rows in the data set) and 178 features (columns).
There are five classes for the data. Quoting from the above link:

y contains the category of the 178-dimensional input vector.
Specifically y in {1, 2, 3, 4, 5}:

    5 - eyes open, means when they were recording the EEG signal of the brain
        the patient had their eyes open
    4 - eyes closed, means when they were recording the EEG signal the patient
        had their eyes closed
    3 - Yes they identify where the region of the tumor was in the brain and
        recording the EEG activity from the healthy brain area
    2 - They recorder the EEG from the area where the tumor was located
    1 - Recording of seizure activity
"""
import itertools
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical, plot_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

LEGEND_COORDS = (1.2, 0.8)
TOTAL_PCA_COMPONENTS = 60
CLASS_MAP = {5: 'eyes open', 4: 'eyes closed', 3: 'healthy brain area',
             2: 'tumor area', 1: 'seizure activity'}


def run(actions):
    """Explore seizure data set and/or create prediction models

    Args:
        actions: list of str
            Action to take. Choose one or more of
              explore - make several plots of the data, including of the
                  Principal Component Analysis (PCA) transformation of
                  the features
              pca_svm2 - train a binary classifier (seizure vs
                  non-seizure) using a pipeline of PCA and a support
                  vector machine
              pca_svm5 - train a multiclass classifier using a pipeline
                  of PCA and a support vector machine
              rf - train a multiclass classifier using a random decision
                   forest
              nn - train a multiclass classifier using an artificial
                   neural network
          All the training methods save the models to the model
          directory and additionally save a confusion matrix to the
          outputs directory.

    Returns:
        None
    """
    epidata = pd.read_csv(os.path.join('data', 'data.csv'))
    set_matplotlib_params()
    # explore_data(epidata)
    features = epidata.drop(['y', 'Unnamed: 0'], axis=1) / 2047.0
    target = epidata['y']
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.3,
                                                        random_state=0)
    if 'explore' in actions:
        run_explore(epidata, x_train, y_train)
    if 'pca_svm2' in actions:
        run_pca_svm2(x_train, y_train, x_test, y_test)
    if 'pca_svm5' in actions:
        run_pca_svm5(x_train, y_train, x_test, y_test)
    if 'rf' in actions:
        run_rf(x_train, y_train, x_test, y_test)
    if 'nn' in actions:
        run_nn(x_train, y_train, x_test, y_test)

def run_explore(epidata, x_train, y_train):
    """Generate several plots to help understand the data

    Returns:
        None
    """
    explore_data(epidata)
    explore_pca(x_train, y_train)
    naive_vis(x_train, y_train)


def run_pca_svm2(x_train, y_train, x_test, y_test):
    """Train binary classifier with PCA and SVM

    Create a pipeline of Principal Component Analysis followed by a kernel
    Support Vector Machine with a radial-basis-function kernel. Train it to
    classify seizures vs non-seizures. Run a cross-validation grid search
    and save the results. Train a new model with all the training data and
    save the model to the models directory. Run the model on the test data
    and save the confusion matrix

    Args:
        x_train: pandas DataFrame
            Features for training
        y_train: pandas Series
            Targets for training
        x_test: pandas DataFrame
            Features for testing
        y_test:
            Targets for testing

    Returns:
        None
    """
    test_pca_svm(x_train, (y_train == 1).astype(int), x_test,
                 (y_test == 1).astype(int), '2c_scaled_fine')
    train_and_save_pca_svm(50, 50, 0.005, x_train, (y_train == 1).astype(int),
                           x_test, (y_test == 1).astype(int),
                           'two_class_pca_svm')

def run_pca_svm5(x_train, y_train, x_test, y_test):
    """Train multiclass classifier with PCA and SVM

    Create a pipeline of Principal Component Analysis followed by a kernel
    Support Vector Machine with a radial-basis-function kernel. Train it to
    classify for the 5-class problem. Run a cross-validation grid search
    and save the results. Train a new model with all the training data and
    save the model to the models directory. Run the model on the test data
    and save the confusion matrix

    Args:
        x_train: pandas DataFrame
            Features for training
        y_train: pandas Series
            Targets for training
        x_test: pandas DataFrame
            Features for testing
        y_test:
            Targets for testing

    Returns:
        None
    """
    test_pca_svm(x_train, y_train, x_test, y_test, '5c_scaled')
    train_and_save_pca_svm(50, 100, 0.1, x_train, y_train, x_test, y_test,
                           'five_class_pca_svm')
    visualize_confusion(os.path.join('outputs', 'five_class_pca_svm'))


def run_rf(x_train, y_train, x_test, y_test):
    """Train multiclass classifier with random decision forest

    Create a random decision forest classifier. Train it to classify for the
    5-class problem. Run a cross-validation grid search and save the results.
    Train a new model with all the training data and save the model to the
    models directory. Run the model on the test data and save the confusion
    matrix

    Args:
        x_train: pandas DataFrame
            Features for training
        y_train: pandas Series
            Targets for training
        x_test: pandas DataFrame
            Features for testing
        y_test:
            Targets for testing

    Returns:
        None
    """
    test_random_forest(x_train, y_train, x_test, y_test, '5c_rf_scaled')
    visualize_confusion(os.path.join('outputs', 'confusion_5c_rf_scaled'))


def run_nn(x_train, y_train, x_test, y_test):
    """Train multiclass classifier with artificial neural network

    Create a classifier using an artificial neural network. Train it to
    classify for the 5-class problem. Save the model to the models directory.
    Run the model on the test data and save the confusion matrix

    Args:
        x_train: pandas DataFrame
            Features for training
        y_train: pandas Series
            Targets for training
        x_test: pandas DataFrame
            Features for testing
        y_test:
            Targets for testing

    Returns:
        None
    """
    create_and_test_neural_net(x_train, x_test, y_train, y_test)
    visualize_confusion(os.path.join('outputs', 'confusion_nn'))

def save_data_to_file(features, targets, filename):
    """Save features and targets to a csv file

    Args:
        features: numpy ndarray
            Features to be used for prediction
        targets: numpy ndarray, one-dimensional
            Targets of prediction. Should have one dimension the same as the
            number of rows in features and the other dimension should be 1
        filename: str
            Path to the comma-separated-value file to save

    Returns:
        None
    """
    to_write = pd.DataFrame(data=features)
    to_write['y'] = targets
    to_write.to_csv(filename)


def visualize_confusion(filename):
    """Make a heatmap of a confusion matrix

    Args:
        filename: string
            Path to a csv file (without extension). Comma-separated-value
            file containing a confusion matrix. Also used for
            the png
            file in which to write out the image.

    Returns:
        None
    """
    conf = pd.read_csv(filename + '.csv', index_col=0)
    conf = conf.drop('All', axis=0)
    conf = conf.drop('All', axis=1)
    plot_axes = sns.heatmap(conf, annot=True, fmt="d")
    plot_axes.set_xlabel('Predicted')
    plot_axes.set_ylabel('Actual')
    fig = plot_axes.get_figure()
    fig.savefig(filename + '.png', bbox_inches='tight')
    fig.clf()


def test_random_forest(x_train, y_train, x_test, y_test, filename_root):
    """Train and test a random forest classifier

    Args:
        x_train: pandas DataFrame
            Features for training
        y_train: pandas Series
            Targets for training
        x_test: pandas DataFrame
            Features for testing
        y_test:
            Targets for testing
        filename_root: str
            Identifier for this set of parameters and targets, to be used for
            writing out the cross-validation results and confusion matrix

    Returns:
        None
    """
    parameters = {
        'max_features': [10, 20, 30],
        'n_estimators': [150, 200, 250],
        'max_depth': [15, 20, 25],
        'random_state': [0],
    }
    clf = GridSearchCV(RandomForestClassifier(), parameters, cv=10, verbose=10)
    clf.fit(x_train, y_train)
    rf_model = clf.best_estimator_
    joblib.dump(rf_model, os.path.join('models', filename_root + '.z'))
    y_pred = rf_model.predict(x_test)
    cv_res = pd.DataFrame(clf.cv_results_)
    cv_res.to_csv(os.path.join('outputs',
                               'cv_results_' + filename_root + '.csv'))
    confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'],
                            colnames=['Predicted'], margins=True)
    confusion.to_csv(os.path.join('outputs',
                                  'confusion_' + filename_root + '.csv'))


def naive_vis(x_train, y_train):
    """Create some plots of some simple summary statistics of the data

    Args:
        x_train: pandas DataFrame
            Features for training
        y_train: pandas Series
            Targets for training

    Returns:
        None
    """
    naive = get_naive_features(x_train)
    make_feature_scatter_plots(naive, y_train)
    make_violin_plots(naive, y_train)


def test_pca_svm(x_train, y_train, x_test, y_test, filename_root):
    """Train and test a PCA and SVM classifier

    Train a pipeline with principal components analysis dimensionality
    reduction followed by a support vector machine (SVM) with a radial basis
    function kernel. Optimize the parameters of the SVM and train with the
    training set. Run on the test set and save the classification report as
    a csv file.

    Args:
        x_train: pandas DataFrame
            Features for training
        y_train: pandas Series
            Targets for training
        x_test: pandas DataFrame
            Features for testing
        y_test:
            Targets for testing
        filename_root: str
            Identifier for this set of parameters and targets, to be used for
            writing out the cross-validation results and confusion matrix

    Returns:
        None
    """
    pca = PCA(svd_solver='randomized', whiten=True)
    svm = SVC(kernel='rbf', class_weight='balanced')
    pipeline = Pipeline(steps=[('pca', pca), ('svm', svm)])
    param_grid = {'svm__C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'svm__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                  'pca__n_components': [2, 5, 10, 20, 30, 40, 50]}
    if len(y_train.unique()) > 2:
        scoring = {'accuracy': 'accuracy',
                   'f1': 'f1_macro', 'precision': 'precision_macro', 'recall':
                       'recall_macro'}
    else:
        scoring = {'AUC': 'roc_auc', 'accuracy': 'accuracy',
                   'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
    clf = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring,
                       refit='f1')
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    cv_res = pd.DataFrame(clf.cv_results_)
    cv_res.to_csv(os.path.join('outputs',
                               'cv_results_' + filename_root + '.csv'))
    confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'],
                            colnames=['Predicted'], margins=True)
    confusion.to_csv(os.path.join('outputs',
                                  'confusion_' + filename_root + '.csv'))


def train_and_save_pca_svm(n_components, C, gamma, x_train, y_train,
                           x_test, y_test, filename_root):
    """Train a pipeline with PCA and RBF SVM and save it to a file for later use

    Build a pipeline with two components: A principal components analysis
    followed by a support-vector machine classifier. Fit the pipeline to
    x_test and y_test. Save the model as a file to be loaded later for
    prediction. Test the prediction using x_test and y_test and save the
    resulting confusion matrix to a file.

    Args:
        n_components: int
            Number of principal components to use in the PCA portion of the
            pipeline
        C: float
            Penalty parameter C of the error term of the SVM classifier
        gamma : float, optional (default=’auto’)
            Kernel coefficient for the radial basis function kernel of the
            SVM classifier
        x_train: pandas DataFrame or numpy ndarray
            Features to use to train the pipeline
        y_train: pandas Series or numpy ndarray
            Targets to use to train the pipeline
        x_test: pandas DataFrame or numpy ndarray
            Features to use to test the pipeline
        y_test: pandas Series or numpy ndarray
            Targets to use to test the pipeline
        filename_root: str
            Identifier for this model, used in creating file names for the
            model file and the confusion matrix file

    Returns:

    """
    pca = PCA(n_components=n_components, whiten=True)
    svm = SVC(kernel='rbf', C=C, gamma=gamma, class_weight='balanced')
    pipeline = Pipeline(steps=[('pca', pca), ('svm', svm)])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'],
                            colnames=['Predicted'], margins=True)
    confusion.to_csv(os.path.join('outputs',
                                  'confusion_' + filename_root + '.csv'))
    model_filename = os.path.join('models', filename_root + '.z')
    joblib.dump(pipeline, model_filename)


def make_violin_plots(features, targets):
    """Make seaborn violin plots of the features, grouped by target value

    For each feature, group the data points by target value. Create a seaborn
    violin plot showing approximate distributions for the feature in each
    target group. Label the plot and save it with a name that references the
    feature.

    Args:
        features: pandas DataFrame
            Feature values. Each row is a data point and each column is a
            feature.
        targets: pandas Series
            Class for each data point. Must have the same number of rows as the
            features argument.

    Returns:
        None
    """
    features['y'] = targets.map(CLASS_MAP)
    for col in features.columns:
        if col is not 'y':
            plot_axes = sns.violinplot(x=col, y='y', data=features)
            fig = plot_axes.get_figure()
            filename = os.path.join('outputs', col + '_violin.png')
            fig.savefig(filename, bbox_inches='tight')
            fig.clf()


def make_feature_scatter_plots(features, targets):
    """Create scatter plots of all 2-combinations of features

    The scatter plots colour the points by class. They are saved to png
    files with the names of the two plotted features.

    Args:
        features: pandas DataFrame
            Features to plot. Rows are data points and columns are features.
        targets: pandas Series
            Classification of each data point. Must have the same number of rows
            as the features argument.

    Returns:
        None
    """
    for (xvar, yvar) in itertools.combinations(features.columns, 2):
        plot_axes = plt.subplot(1, 1, 1)
        for class_num in sorted(targets.unique()):
            to_plot = targets == class_num
            xvals = features[xvar][to_plot]
            yvals = features[yvar][to_plot]
            plot_axes.scatter(xvals, yvals, label=CLASS_MAP[class_num], s=1)
        plot_axes.set_xlabel(xvar)
        plot_axes.set_ylabel(yvar)
        plot_axes.legend(loc='right', bbox_to_anchor=(1.4, 0.8))
        fig = plot_axes.get_figure()
        filename = os.path.join('outputs', xvar + '_vs_' + yvar + '.png')
        fig.savefig(filename, bbox_inches='tight')
        fig.clf()


def get_naive_features(data):
    """Calculate the mean, min, max, std, quartiles, and range of the data

    Args:
        data: pandas DataFrame
            data to calculate statistics of. Each data point is a row.

    Returns: pandas DataFrame
        Columns are 'min', 'max', 'range', 'mean', '25%', '50%', and '75%'
    """
    result = data.transpose().describe().transpose()
    result = result.drop('count', axis=1)
    result['range'] = result['max'] - result['min']
    return result


def explore_pca(x_train, y_train, n_components=TOTAL_PCA_COMPONENTS):
    """Create plots of Principal Component Analysis decomposition

    Find the first TOTAL_PCA_COMPONENTS PCA components of the argument. Create
    a plot of the explained variance ratio and a plot of the cumulative sum of
    the explained variance ratio, over the number of PCA components used. Save
    the explained variance ratio to a comma-separated file. Create a scatter
    plot of the data points versus the first two principal components, coloured
    by classification target. Create a line plot of the first five principal
    components.

    Args:
        x_train: pandas.DataFrame
            Training data to decompose using Principal Component Analysis
        y_train: pandas.Series
            Classification targets
        n_components : int, Optional, default 60
            Number of PCA components to use in plots and csv file

    Returns:
        None
    """
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(x_train)
    components = pca.components_
    var_ratio = pd.Series(pca.explained_variance_ratio_)
    sum_var_ratio = var_ratio.cumsum()
    var_ratio.to_csv(os.path.join('outputs', 'var_ratio.csv'), header=True)
    plot_explained_variance(var_ratio, os.path.join('outputs', 'var_ratio.png'))
    plot_sum_explained_variance(sum_var_ratio,
                                os.path.join('outputs', 'var_ratio_sum.png'))
    first_two_pca_scatter(transformed[:, :2], y_train,
                          os.path.join('outputs', 'two_pca_components.png'))
    plot_components(components[:3, :], os.path.join('outputs',
                                                    'pca_components.png'))


def plot_components(components, filename):
    """Plot PCA components

    Args:
        components: pandas DataFrame or numpy ndarray
            Each component is a row and each column is one of the original
            features
        filename: str
            Path to file to which to save the figure. File format is inferred
            from the extension

    Returns:
        None
    """
    plot_axes = plt.subplot(1, 1, 1)
    plot_axes.plot(components.transpose())
    plot_axes.set_xlabel('Feature')
    plot_axes.set_ylabel('Component values')
    fig = plot_axes.get_figure()
    fig.savefig(filename, bbox_inches='tight')
    fig.clf()


def first_two_pca_scatter(components, targets, filename):
    """Make scatter plot of data vs 1st 2 PCA components, colored by target

    Args:
        components: numpy ndarray
            Principal Component Analysis components. Each row is a data point
            and each column is the value of that PCA component for that
            data point.
        targets: numpy ndarray or pandas Series
            Labels for each data point. Must have the same number of rows as
            the components argument
        filename: str
            Path to file to which to save the figure. File format is inferred
            from the extension

    Returns:
        None
    """
    plot_axes = plt.subplot(1, 1, 1)
    for class_num in sorted(targets.unique()):
        to_plot = targets == class_num
        plot_axes.scatter(components[to_plot, 0], components[to_plot, 1],
                          label=CLASS_MAP[class_num], s=1)
    plot_axes.set_xlabel('PCA component 1')
    plot_axes.set_ylabel('PCA component 2')
    plot_axes.legend(loc='right', bbox_to_anchor=(1.4, 0.8))
    fig = plot_axes.get_figure()
    fig.savefig(filename, bbox_inches='tight')
    fig.clf()


def plot_sum_explained_variance(var_ratio, filename):
    """Plot one minus cumulative sum of the explained variance of a transform

    Create a plot with a linear x-axis and a logarithmic y-axis of one minus
    the cumulative explained variance of the components of a dimensionality
    reduction method and save the plot to a file. This plot allows the user
    to determine how many components are required to achieve a certain total
    explained variance ratio, such as 90% or 99%.

    Args:
        var_ratio: pandas.Series
            Explained variance ratio of each component. Should be
            monotonically decreasing.
        filename: str
            Path to file to which to save the figure. File format is inferred
            from the extension
    Returns:
        None
    """
    plot_axes = plt.subplot(1, 1, 1)
    plot_axes.semilogy(1 - var_ratio, marker='.', linestyle='none')
    plot_axes.set_xlabel('PCA component')
    plot_axes.set_ylabel('1 - sum(Explained variance ratio)')
    fig = plot_axes.get_figure()
    fig.savefig(filename)
    fig.clf()


def plot_explained_variance(var_ratio, filename):
    """Plot the explained variance of each component of a transform

    Create a plot with a linear x-axis and a logarithmic y-axis of the
    explained variance ratio of each component. This plot allows the user
    to see how rapidly the explained variance decreases by component and to
    identify and eigengap, if it exists.

    Args:
        var_ratio: pandas.Series
            Explained variance ratio of each component. Should be
            monotonically decreasing.
        filename: str
            Path to file to which to save the figure. File format is inferred
            from the extension

    Returns:
        None
    """
    plot_axes = plt.subplot(1, 1, 1)
    plot_axes.semilogy(var_ratio, marker='.', linestyle='none')
    plot_axes.set_xlabel('PCA component')
    plot_axes.set_ylabel('Explained variance ratio')
    fig = plot_axes.get_figure()
    fig.savefig(filename)
    fig.clf()


def explore_data(epidata):
    """Create a number of plots and csv files to explore the seizure data

    Args:
        epidata: pandas.DataFrame
            Data about seizures. Each row is a data point and each column is
            a feature, except for the column 'y' which contains the
            classification target.
    Returns:
        None
    """
    features = epidata.drop(['y', 'Unnamed: 0'], axis=1)
    desc = features.describe()
    desc.transpose().describe().to_csv(os.path.join('outputs',
                                                    'double_desc.csv'))
    epidata['y'].value_counts().to_csv(os.path.join('outputs',
                                                    'class_counts.csv'),
                                       header=True)
    create_summary_plot(desc, os.path.join('outputs', 'feature_summary.png'))
    create_interquartile_plot(desc, os.path.join('outputs',
                                                 'interquartile.png'))
    create_mean_median_plot(desc, os.path.join('outputs', 'mean_median.png'))
    create_std_plot(desc.loc['std', :], os.path.join('outputs',
                                                     'feature_std.png'))
    create_corr_heatmap(features, os.path.join('outputs', 'corr_heatmap.png'),
                        os.path.join('outputs', 'corr_X90.png'))


def create_corr_heatmap(features, filename1, filename2):
    """Create 2 plots: a feature correlation heat map and correlations with X90

    Args:
        features: pandas.DataFrame
            Feature columns. Each row is a data point and each column is a
            feature. One column must be named 'X90'.
        filename1: str
            Path to which to save correlation heatmap figure. File format is
            inferred from the extension.
        filename2:
            Path to which to save figure showing correlations with feature X90.
            File format is inferred from the extension.

    Returns:
        None
    """
    corr_mat = features.corr()
    corr_mat.to_csv(os.path.join('outputs', 'corr_mat.csv'))
    plot_axes = corr_mat['X90'].plot()
    plot_axes.set_xlabel('Feature')
    plot_axes.set_ylabel('Correlation with feature X90')
    fig = plot_axes.get_figure()
    fig.savefig(filename2)
    fig.clf()
    sns.heatmap(corr_mat, center=0, cmap='coolwarm')
    fig.savefig(filename1)
    fig.clf()


def create_std_plot(std, filename):
    """Create plot of standard deviation of each feature

    Args:
        std: pandas.Series
            Standard deviation of each feature across the data set. The index is
            the name of the feature.
        filename: str
            Path to which to save the figure. The file format is inferred
            from the extension.

    Returns:
        None
    """
    plot_axes = std.plot()
    plot_axes.set_xlabel('Feature')
    plot_axes.set_ylabel('Standard deviation of feature value')
    fig = plot_axes.get_figure()
    fig.savefig(filename)
    fig.clf()


def create_summary_plot(description, filename):
    """Create a plot showing mean, min, max, and quartiles of features

    Args:
        description: pandas.DataFrame
            Description of the features. Result of running the
            pandas.DataFrame.describe method on the features
        filename: str
            Path to which to save the figure. The file format is inferred
            from the extension.

    Returns:
        None
    """
    to_plot = description.drop(['count', 'std']).transpose()
    create_feature_value_plot(to_plot, filename)


def create_interquartile_plot(data, filename):
    """Create a plot of the mean, median, and 25th and 75th percentile

    Args:
        data:  pandas.DataFrame
            Description of the features. Result of running the
            pandas.DataFrame.describe method on the features
        filename: str
            Path to which to save the figure. The file format is inferred
            from the extension.

    Returns:
        None
    """
    cols = ['mean', '25%', '50%', '75%']
    to_plot = data.transpose()[cols]
    create_feature_value_plot(to_plot, filename)


def create_mean_median_plot(data, filename):
    """Create a plot of the mean and median of the features

        Args:
            data:  pandas.DataFrame
                Description of the features. Result of running the
                pandas.DataFrame.describe method on the features
            filename: str
                Path to which to save the figure. The file format is inferred
                from the extension.

        Returns:
            None
        """
    cols = ['mean', '50%']
    to_plot = data.transpose()[cols]
    to_plot.columns = ['mean', 'median']
    create_feature_value_plot(to_plot, filename)


def create_feature_value_plot(data, filename):
    """Create a plot with features on the x-axis and values on the y-axis

        Args:
            data:  pandas.DataFrame
                Data to plot
            filename: str
                Path to which to save the figure. The file format is inferred
                from the extension.

        Returns:
            None
        """
    plot_axes = data.plot()
    plot_axes.set_xlabel('Feature')
    plot_axes.set_ylabel('Value')
    plot_axes.legend(loc='right', bbox_to_anchor=LEGEND_COORDS)
    fig = plot_axes.get_figure()
    fig.savefig(filename, bbox_inches='tight')
    fig.clf()


def set_matplotlib_params():
    """Set matplotlib parameters to chosen aesthetics

    Set the figure dots per inch to 200, and the edge, tick and axis label
    colors to gray for all future matplotlib calls.

    Returns:
        None
    """
    matplotlib.rcParams['savefig.dpi'] = 200
    matplotlib.rcParams['axes.edgecolor'] = 'gray'
    matplotlib.rcParams['xtick.color'] = 'gray'
    matplotlib.rcParams['ytick.color'] = 'gray'
    matplotlib.rcParams['axes.labelcolor'] = 'gray'


def create_and_test_neural_net(x_train, x_test, y_train, y_test):
    """Create a neural network for the epileptic seizure classification data set

    Train a neural network for multiclass classification. Print out the
    accuracy. Save the model to disk for later use in prediction. Save a
    visualization of the model.

    Returns:
        None
    """
    model = train_nn(x_train, y_train)
    test_nn(model, x_test, y_test)
    model.save('neural_net_5c.h5')
    plot_model(model, show_shapes=True,
               to_file=os.path.join('outputs', 'nn_model.png'))


def test_nn(model, x_test, y_test):
    """Test the neural network on the test set and print the accuracy

        Args:
            model: keras neural network model
                Neural network to use for prediction
            x_test: pandas DataFrame
                Test features
            y_test: binary class matrix
                Test classes. To get the right format, call
                keras.util.to_categorical on e.g. a pandas Series of integers.

        Returns:
            None
    """
    y_test2 = to_categorical(y_test - 1, num_classes=5)
    loss_and_metrics = model.evaluate(x_test, y_test2)
    print(model.metrics_names)
    print(loss_and_metrics)
    y_pred = model.predict(x_test)
    y_pred2 = y_pred.argmax(axis=1)
    confusion = pd.crosstab(y_test, y_pred2, rownames=['Actual'],
                            colnames=['Predicted'], margins=True)
    confusion.to_csv(os.path.join('outputs', 'confusion_nn' + '.csv'))


def train_nn(x_train, y_train):
    """Train a neural network

        Args:
            x_train: pandas DataFrame
                Training features
            y_train: binary class matrix
                Training classes. To get the right format, call
                keras.util.to_categorical on e.g. a pandas Series of integers.

        Returns:
            None
    """
    y_train2 = to_categorical(y_train.values - 1, num_classes=5)
    model = Sequential()
    model.add(Dense(units=178, input_shape=(178,), activation='relu'))
    model.add(Dense(units=80, activation='relu'))
    model.add(Dense(units=40, activation='relu'))
    model.add(Dense(units=5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd',
                  metrics=['accuracy'])
    # history = model.fit(x_train, y_train, validation_split=0.2, epochs=200)
    history = model.fit(x_train, y_train2, epochs=150)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join('outputs', 'train_history.csv'))
    filename = os.path.join('models', '5c_nn.h5')
    model.save(filename)
    return model


if __name__ == '__main__':
    print('To run methods in this module, use the run_epiclass.py script')