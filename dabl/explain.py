from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from .models import SimpleClassifier
from .plotting import plot_coefficients


def explain(estimator, feature_names=None):
    if feature_names is None:
        try:
            feature_names = estimator.feature_names_
        except AttributeError:
            raise ValueError("Can't determine input feature names, "
                             "please pass them.")

    # Start unpacking the estimator to get to the final step
    if isinstance(estimator, SimpleClassifier):
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
        # FIXME !!! ug in plot_tree with integer class names
        class_names = [str(c) for c in estimator.classes_]
        plot_tree(estimator, feature_names=feature_names,
                  class_names=class_names, filled=True)
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
