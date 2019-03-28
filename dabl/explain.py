import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from .models import EasyClassifier


def plot_coefficients(coefficients, feature_names, n_top_features=25):
    """Visualize coefficients of a linear model.

    Parameters
    ----------
    coefficients : nd-array, shape (n_features,)
        Model coefficients.

    feature_names : list or nd-array of strings, shape (n_features,)
        Feature names for labeling the coefficients.

    n_top_features : int, default=25
        How many features to show. The function will show the largest (most
        positive) and smallest (most negative)  n_top_features coefficients,
        for a total of 2 * n_top_features coefficients.
    """
    coefficients = coefficients.squeeze()
    if coefficients.ndim > 1:
        # this is not a row or column vector
        raise ValueError("coeffients must be 1d array or column vector, got"
                         " shape {}".format(coefficients.shape))
    coefficients = coefficients.ravel()

    if len(coefficients) != len(feature_names):
        raise ValueError("Number of coefficients {} doesn't match number of"
                         "feature names {}.".format(len(coefficients),
                                                    len(feature_names)))
    # get coefficients with large absolute values
    coef = coefficients.ravel()
    positive_coefficients = np.argsort(coef)[-n_top_features:]
    negative_coefficients = np.argsort(coef)[:n_top_features]
    interesting_coefficients = np.hstack([negative_coefficients,
                                          positive_coefficients])
    # plot them
    plt.figure(figsize=(15, 5))
    colors = ['red' if c < 0 else 'blue'
              for c in coef[interesting_coefficients]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefficients],
            color=colors)
    feature_names = np.array(feature_names)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features),
               feature_names[interesting_coefficients], rotation=60,
               ha="right")
    plt.ylabel("Coefficient magnitude")
    plt.xlabel("Feature")


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img


def explain(estimator, feature_names=None):
    if feature_names is None:
        try:
            feature_names = estimator.feature_names_
        except AttributeError:
            raise ValueError("Can't determine input feature names, "
                             "please pass them.")

    # Start unpacking the estimator to get to the final step
    if isinstance(estimator, EasyClassifier):
        # get the pipeline
        estimator = estimator.est_
        # pipelines don't have feature names yet in sklearn
        # *cries in scikit-learn roadmap*
        final_est = estimator[-1]
        try:
            feature_names = estimator[0].get_feature_names(feature_names)
        except TypeError:
            feature_names = estimator[0].get_feature_names()

        # now we have input feature names for the final step
        estimator = final_est

    if isinstance(estimator, DecisionTreeClassifier):
        print(estimator)
        print("Depth: {}".format(estimator.get_depth()))
        print("Number of leaves: {}".format(estimator.get_n_leaves()))
        plot_tree(estimator, feature_names=feature_names,
                  class_names=estimator.classes_, filled=True)
        # FIXME This is a bad thing to show!
        plot_coefficients(estimator.feature_importances_, feature_names)
    elif hasattr(estimator, 'coef_'):
        # probably a linear model, can definitely show the coefficients
        plot_coefficients(estimator.coef_, feature_names)
    elif isinstance(estimator, RandomForestClassifier):
        # FIXME This is a bad thing to show!

        plot_coefficients(estimator.feature_importances_, feature_names)

    else:
        raise ValueError("Don't know how to explain estimator {} "
                         "yet.".format(estimator))
