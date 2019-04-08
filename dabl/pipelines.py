from warnings import warn

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier


def get_fast_classifiers(n_classes):
    """Get a list of very fast classifiers.

    Parameters
    ----------
    n_classes : int
        Number of classes in the dataset. Used to decide on the complexity
        of some of the classifiers.


    Returns
    -------
    fast_classifiers : list of sklearn estimators
        List of classification models that can be fitted and evaluated very
        quickly.
    """
    return [
        # These are sorted by approximate speed
        DummyClassifier(strategy="prior"),
        GaussianNB(),
        make_pipeline(MinMaxScaler(), MultinomialNB()),
        DecisionTreeClassifier(max_depth=1, class_weight="balanced"),
        DecisionTreeClassifier(max_depth=max(5, n_classes),
                               class_weight="balanced"),
        DecisionTreeClassifier(class_weight="balanced",
                               min_impurity_decrease=.01),
        LogisticRegression(C=.1, solver='lbfgs', multi_class='auto',
                           class_weight='balanced')
    ]


def get_fast_regressors():
    """Get a list of very fast regressors.

    Returns
    -------
    fast_regressors : list of sklearn estimators
        List of regression models that can be fitted and evaluated very
        quickly.
    """
    return [
        DummyRegressor(),
        DecisionTreeRegressor(max_depth=1),
        DecisionTreeRegressor(max_depth=5),
        Ridge(alpha=10),
        Lasso(alpha=10)]


def get_any_classifiers():
    sklearn_ests = [
        LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=.1, solver='lbfgs', multi_class='multinomial'),
        RandomForestClassifier(max_features=None, n_estimators=100),
        RandomForestClassifier(max_features='sqrt', n_estimators=100),
        RandomForestClassifier(max_features='log2', n_estimators=100)]
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        warn("lightgbm not installed, skipping gradient boosting models",
             UserWarning)
        return sklearn_ests

    gb_ests = [
        LGBMClassifier(learning_rate=0.2),
        LGBMClassifier(learning_rate=0.1),
        LGBMClassifier(learning_rate=0.01),
        LGBMClassifier(learning_rate=0.001),
        LGBMClassifier(learning_rate=0.2, colsample_bytree=.5),
        LGBMClassifier(learning_rate=0.1, colsample_bytree=.5),
        LGBMClassifier(learning_rate=0.01, colsample_bytree=.5),
        LGBMClassifier(learning_rate=0.001, colsample_bytree=.5),
        ]
    return sklearn_ests + gb_ests
