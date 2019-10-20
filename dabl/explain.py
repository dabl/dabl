import numpy as np
from warnings import warn

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import plot_partial_dependence

from .models import SimpleClassifier, SimpleRegressor, AnyClassifier
from .utils import nice_repr, _validate_Xyt
from .plot.utils import (plot_coefficients, plot_multiclass_roc_curve,
                         find_pretty_grid)


def classification_metrics(estimator, X_val, y_val):
    y_pred = estimator.predict(X_val)
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    try:
        from sklearn.metrics import plot_roc_curve
        if len(estimator.classes_) == 2:
            plot_roc_curve(estimator, X_val, y_val)
        elif len(estimator.classes_) > 2:
            plot_multiclass_roc_curve(estimator, X_val, y_val)
    except ImportError:
        warn("Can't plot roc curve, install sklearn 0.22-dev")


def explain(estimator, X_val=None, y_val=None, target_col=None,
            feature_names=None):
    """Explain estimator.

    Provide basic properties and evaluation plots for the estimator.

    Parameters
    ----------
    estimator : dabl or sklearn estimator
        Model to evaluate.

    X_val : DataFrame, optional
        Validation set. Used for computing hold-out evaluations
        like roc-curves, permutation importance or partial dependence plots.

    y_val : Series or numpy array, optional.
        Validation set labels. You need to specify either y_val or target_col.

    target_col : string or int, optional
        Column name of target if included in X.
    """
    if feature_names is None:
        try:
            feature_names = estimator.feature_names_.to_list()
        except AttributeError:
            raise ValueError("Can't determine input feature names, "
                             "please pass them.")
    classifier = False
    if hasattr(estimator, 'classes_') and len(estimator.classes_) >= 2:
        n_classes = len(estimator.classes_)
        classifier = True
    else:
        n_classes = 1

    if X_val is not None:
        X_val, y_val = _validate_Xyt(X_val, y_val, target_col)

        if classifier:
            # classification metrics:
            classification_metrics(estimator, X_val, y_val)
        else:
            # FIXME
            pass
        # FIXME Skip for linear models?
        n_rows, n_cols = find_pretty_grid(len(feature_names))
        print("Computing partial dependence plots...")
        if n_classes <= 2:
            plot_partial_dependence(estimator, X_val, features=feature_names,
                                    feature_names=feature_names, n_cols=n_cols)
        else:
            for c in estimator.classes_:
                plot_partial_dependence(estimator, X_val,
                                        features=feature_names,
                                        feature_names=feature_names,
                                        target=c, n_cols=n_cols)

    # Start unpacking the estimator to get to the final step
    inner_estimator = estimator
    if (isinstance(inner_estimator, SimpleClassifier)
            or isinstance(inner_estimator, SimpleRegressor)):
        # get the pipeline
        inner_estimator = inner_estimator.est_
    elif isinstance(inner_estimator, AnyClassifier):
        inner_estimator = inner_estimator.est_
    if isinstance(inner_estimator, Pipeline):
        assert len(inner_estimator.steps) == 2
        # pipelines don't have feature names yet in sklearn
        # *cries in scikit-learn roadmap*
        final_est = inner_estimator._final_estimator
        try:
            feature_names = inner_estimator.steps[0][1].get_feature_names(
                feature_names)
        except TypeError:
            feature_names = inner_estimator.steps[0][1].get_feature_names()

        # now we have input feature names for the final step
        inner_estimator = final_est
    # done unwrapping, start evaluating

    if isinstance(inner_estimator, DecisionTreeClassifier):
        print(nice_repr(inner_estimator))
        try:
            print("Depth: {}".format(inner_estimator.get_depth()))
            print("Number of leaves: {}".format(
                inner_estimator.get_n_leaves()))
        except AttributeError:
            warn("Can't show tree depth, install scikit-learn 0.21"
                 " to show the full information.")
        # FIXME !!! bug in plot_tree with integer class names
        class_names = [str(c) for c in estimator.classes_]
        plt.figure(figsize=(18, 10))
        plot_tree(inner_estimator, feature_names=feature_names,
                  class_names=class_names, filled=True, max_depth=5,
                  precision=2, proportion=True)
        # FIXME This is a bad thing to show!
        plot_coefficients(inner_estimator.feature_importances_, feature_names)
        plt.ylabel("Impurity Decrease")
    elif hasattr(estimator, 'coef_'):
        # probably a linear model, can definitely show the coefficients
        # would be nice to have the target name here
        if hasattr(inner_estimator, "classes_"):
            coef = np.atleast_2d(inner_estimator.coef_)
            for k, c in zip(inner_estimator.classes_, coef):
                plot_coefficients(c, feature_names,
                                  classname="class: {}".format(k))
        else:
            coef = np.squeeze(inner_estimator.coef_)
            if coef.ndim > 1:
                raise ValueError("Don't know how to handle "
                                 "multi-target regressor")
            plot_coefficients(coef, feature_names)
    elif isinstance(inner_estimator, RandomForestClassifier):
        # FIXME This is a bad thing to show!

        plot_coefficients(inner_estimator.feature_importances_, feature_names)
        plt.ylabel("Imputity Decrease")

    else:
        raise ValueError("Don't know how to explain estimator {} "
                         "yet.".format(inner_estimator))
