import os

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

LEGEND_COORDS = (1.2, 0.8)
TOTAL_PCA_COMPONENTS = 60


def main():
    """Explore epileptic seizure classification data set

    Returns:
        None
    """
    epidata = pd.read_csv(os.path.join('data', 'data.csv'))
    set_matplotlib_params()
    explore_data(epidata)
    features = epidata.drop(['y', 'Unnamed: 0'], axis=1)
    target = epidata['y']
    x_train, x_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.3,
                                                        random_state=0)
    print(type(x_train))
    explore_pca(x_train)


def explore_pca(x_train, n_components=TOTAL_PCA_COMPONENTS):
    """Create plots of Principal Component Analysis decomposition

    Find the first TOTAL_PCA_COMPONENTS PCA components of the argument. Create
    a plot of the explained variance ratio and a plot of the cumulative sum of
    the explained variance ratio, over the number of PCA components used. Save
    the explained variance ratio to a comma-separated file.

    Args:
        x_train: pandas.DataFrame
            Training data to decompose using Principal Component Analysis
        n_components : int, Optional, default 60
            Number of PCA components to use in plots and csv file

    Returns:
        None
    """
    pca = PCA(n_components=n_components)
    pca.fit(x_train)
    var_ratio = pd.Series(pca.explained_variance_ratio_)
    sum_var_ratio = var_ratio.cumsum()
    var_ratio.to_csv(os.path.join('outputs', 'var_ratio.csv'), header=True)
    plot_explained_variance(var_ratio, os.path.join('outputs', 'var_ratio.png'))
    plot_sum_explained_variance(sum_var_ratio,
                                os.path.join('outputs', 'var_ratio_sum.png'))


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
            File to save. File format is inferred from the extension
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
            File to save. File format is inferred from the extension

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


if __name__ == '__main__':
    main()
