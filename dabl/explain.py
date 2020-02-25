import numpy as np
import pandas as pd
from warnings import warn

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve

from sklearn.inspection import plot_partial_dependence
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer

from .models import SimpleClassifier, SimpleRegressor, AnyClassifier
from .utils import _validate_Xyt
from .plot.utils import (plot_coefficients, plot_multiclass_roc_curve,
                         find_pretty_grid, _make_subplots, _find_inliers,
                         _get_scatter_alpha,
                         _get_scatter_size)


def plot_classification_metrics(estimator, X_val, y_val):
    """Show classification metrics on a validation set.

    Parameters
    ----------
    estimator : sklearn or dabl estimator
        Trained estimator
    X_val : DataFrame
        Validation set features
    y_val : Series
        Validation set target.
    """
    y_pred = estimator.predict(X_val)
    print(classification_report(y_val, y_pred))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    cf = plot_confusion_matrix(estimator, X_val, y_val, ax=ax[0],
                               values_format='d')
    ax[0].images[0].colorbar.remove()

    ax[1].set_title("ROC Curve")
    if len(estimator.classes_) == 2:
        roc = plot_roc_curve(estimator, X_val, y_val, ax=ax[1])
        ax[2].set_title("Precision recall curve")
        pr = plot_precision_recall_curve(estimator, X_val, y_val, ax=ax[2])
    else:
        roc = plot_multiclass_roc_curve(estimator, X_val, y_val, ax=ax[1])
        pr = None
    return cf, roc, pr


def plot_regression_metrics(estimator, X_val, y_val, drop_outliers=False):
    """Show classification metrics on a validation set.

    Parameters
    ----------
    estimator : sklearn or dabl estimator
        Trained estimator
    X_val : DataFrame
        Validation set features
    y_val : Series
        Validation set target.
    drop_outliers : boolean, default=False
        Whether to drop outliers in the predictions.
    """
    y_pred = estimator.predict(X_val)
    # residual plot
    if drop_outliers:
        inlier_mask = _find_inliers(pd.Series(y_pred))
        y_val = y_val[inlier_mask]
        y_pred = y_pred[inlier_mask]
    true_min = y_val.min()
    true_max = y_val.max()
    scatter_alpha = _get_scatter_alpha('auto', y_val)
    scatter_size = _get_scatter_size('auto', y_val)
    plt.plot([true_min, true_max], [true_min, true_max], '--', label='Ideal')
    plt.scatter(y_val, y_pred, alpha=scatter_alpha, s=scatter_size)
    plt.xlabel("Predictions")
    plt.ylabel("True target")
    plt.title("Observed vs predicted")
    plt.legend()
    ax = plt.gca()
    ax.set_aspect("equal")


def explain(estimator, X_val=None, y_val=None, target_col=None,
            feature_names=None, n_top_features=10):
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

    n_top_features : int, default=10
        Number of features to show for coefficients or feature importances.

    """
    figures = []
    if feature_names is None:
        try:
            feature_names = estimator.feature_names_.to_list()
        except AttributeError:
            if hasattr(X_val, 'columns'):
                feature_names = X_val.columns
            else:
                raise ValueError("Can't determine input feature names, "
                                 "please pass them.")

    classifier = False
    if hasattr(estimator, 'classes_') and len(estimator.classes_) >= 2:
        n_classes = len(estimator.classes_)
        classifier = True
    else:
        n_classes = 1

    inner_estimator, inner_feature_names = _extract_inner_estimator(
        estimator, feature_names)

    if X_val is not None:
        X_val, y_val = _validate_Xyt(X_val, y_val, target_col,
                                     do_clean=False)

        if classifier:
            fig = plot_classification_metrics(estimator, X_val, y_val)
        else:
            fig = plot_regression_metrics(estimator, X_val, y_val)
        figures.append(fig)

    if isinstance(inner_estimator, DecisionTreeClassifier):
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
        plot_tree(inner_estimator, feature_names=inner_feature_names,
                  class_names=class_names, filled=True, max_depth=5,
                  precision=2, proportion=True)
        figures.append(plt.gcf())

        # FIXME This is a bad thing to show!
        plot_coefficients(
            inner_estimator.feature_importances_, inner_feature_names)
        plt.ylabel("Impurity Decrease")
        figures.append(plt.gcf())

    elif hasattr(inner_estimator, 'coef_'):
        # probably a linear model, can definitely show the coefficients
        if n_classes > 2:
            fix, axes = _make_subplots(n_classes)
            coef = np.atleast_2d(inner_estimator.coef_)
            for ax, k, c in zip(axes.ravel(), inner_estimator.classes_, coef):
                plot_coefficients(c, inner_feature_names, ax=ax,
                                  classname="class: {}".format(k),
                                  n_top_features=n_top_features)
        else:
            coef = np.squeeze(inner_estimator.coef_)
            if coef.ndim > 1:
                raise ValueError("Don't know how to handle "
                                 "multi-target regressor")
            plot_coefficients(coef, inner_feature_names,
                              n_top_features=n_top_features)
        figures.append(plt.gcf())

    elif isinstance(inner_estimator, RandomForestClassifier):
        # FIXME This is a bad thing to show!
        plot_coefficients(
            inner_estimator.feature_importances_, inner_feature_names,
            n_top_features=n_top_features)
        plt.ylabel("Imputity Decrease")

    if X_val is not None:
        # feature names might change during preprocessing
        # but we don't want partial dependence plots for one-hot features
        idx, org_features = zip(
            *[(i, f) for i, f in enumerate(inner_feature_names)
              if f in feature_names])
        if hasattr(inner_estimator, 'feature_importances_'):
            importances = inner_estimator.feature_importances_[
                np.array(idx)]
            features = inner_feature_names[np.argsort(importances)[::-1][:10]]
        else:
            X_cont_imputed = SimpleImputer().fit_transform(
                X_val.loc[:, org_features])
            importances, p = f_classif(X_cont_imputed, y_val)
            features = np.array(org_features)[
                np.argsort(importances)[::-1][:10]]
        if not hasattr(inner_estimator, 'coef_'):
            print("Computing partial dependence plots...")
            n_rows, n_cols = find_pretty_grid(len(features))
            try:
                if n_classes <= 2:
                    plot = plot_partial_dependence(
                        estimator, X_val, features=features,
                        feature_names=np.array(feature_names), n_cols=n_cols)
                    plot.figure_.suptitle("Partial Dependence")
                    for ax in plot.axes_.ravel():
                        ax.set_ylabel('')
                else:
                    for c in estimator.classes_:
                        plot = plot_partial_dependence(
                            estimator, X_val, features=features,
                            feature_names=np.array(feature_names),
                            target=c, n_cols=n_cols)
                        plot.figure_.suptitle(
                            "Partial Dependence for class {}".format(c))
                        for ax in plot.axes_.ravel():
                            ax.set_ylabel('')
                figures.append(plot)
            except ValueError as e:
                warn("Couldn't run partial dependence plot: " + str(e))
    return figures


def _extract_inner_estimator(estimator, feature_names):
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
    return inner_estimator, np.array(feature_names, dtype=object)
