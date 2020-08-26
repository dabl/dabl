from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

from .portfolios.portfolio_base import portfolio_base
from .portfolios.portfolio_mixed import portfolio_mixed
from .portfolios.portfolio_hgb import portfolio_hgb
from .portfolios.portfolio_svc import portfolio_svc
from .portfolios.portfolio_rf import portfolio_rf
from .portfolios.portfolio_lr import portfolio_lr

enable_hist_gradient_boosting


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
        LogisticRegression(C=.1,
                           solver='lbfgs',
                           multi_class='auto',
                           class_weight='balanced',
                           max_iter=1000),
        # FIXME Add warm starting here?
        LogisticRegression(C=1,
                           solver='lbfgs',
                           multi_class='auto',
                           class_weight='balanced',
                           max_iter=1000)
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
        Lasso(alpha=10)
    ]


def get_any_classifiers(portfolio='baseline'):
    """Return a portfolio of classifiers.

    Returns
    -------
    classifiers : list of sklearn estimators
        List of classification models.
    """
    baseline = portfolio_base()
    mixed = portfolio_mixed()
    hgb = portfolio_hgb()
    svc = portfolio_svc()
    rf = portfolio_rf()
    lr = portfolio_lr()

    portfolios = {
        'baseline': baseline,
        'mixed': mixed,
        'svc': svc,
        'hgb': hgb,
        'rf': rf,
        'lr': lr
    }

    return (portfolios[portfolio])
