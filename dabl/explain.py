import numpy as np
from warnings import warn

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .models import SimpleClassifier, SimpleRegressor
from .plot.utils import plot_coefficients
from ._plot_tree import plot_tree


def explain(estimator, feature_names=None):
    if feature_names is None:
        try:
            feature_names = estimator.feature_names_
        except AttributeError:
            raise ValueError("Can't determine input feature names, "
                             "please pass them.")

    # Start unpacking the estimator to get to the final step
    if (isinstance(estimator, SimpleClassifier)
            or isinstance(estimator, SimpleRegressor)):
        # get the pipeline
        estimator = estimator.est_
    if isinstance(estimator, Pipeline):
        assert len(estimator.steps) == 2
        # pipelines don't have feature names yet in sklearn
        # *cries in scikit-learn roadmap*
        final_est = estimator._final_estimator
        try:
            feature_names = estimator.steps[0][1].get_feature_names(feature_names)
        except TypeError:
            feature_names = estimator.steps[0][1].get_feature_names()

        # now we have input feature names for the final step
        estimator = final_est

    if isinstance(estimator, DecisionTreeClassifier):
        print(estimator)
        try:
            print("Depth: {}".format(estimator.get_depth()))
            print("Number of leaves: {}".format(estimator.get_n_leaves()))
        except AttributeError:
            warn("Can't show tree depth, install scikit-learn 0.21-dev"
                 " to show the full information.")
        # FIXME !!! bug in plot_tree with integer class names
        class_names = [str(c) for c in estimator.classes_]
        plot_tree(estimator, feature_names=feature_names,
                  class_names=class_names, filled=True, max_depth=5)
        # FIXME This is a bad thing to show!
        plot_coefficients(estimator.feature_importances_, feature_names)
    elif hasattr(estimator, 'coef_'):
        # probably a linear model, can definitely show the coefficients
        # would be nice to have the target name here
        if hasattr(estimator, "classes_"):
            coef = np.atleast_2d(estimator.coef_)
            for k, c in zip(estimator.classes_, coef):
                plot_coefficients(c, feature_names,
                                  classname="class: {}".format(k))
        else:
            coef = np.squeeze(estimator.coef_)
            if coef.ndim > 1:
                raise ValueError("Don't know how to handle "
                                 "multi-target regressor")
            plot_coefficients(coef, feature_names)
    elif isinstance(estimator, RandomForestClassifier):
        # FIXME This is a bad thing to show!

        plot_coefficients(estimator.feature_importances_, feature_names)

    else:
        raise ValueError("Don't know how to explain estimator {} "
                         "yet.".format(estimator))
