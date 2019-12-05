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
        LogisticRegression(C=.1, solver='lbfgs', multi_class='auto',
                           class_weight='balanced', max_iter=1000),
        # FIXME Add warm starting here?
        LogisticRegression(C=1, solver='lbfgs', multi_class='auto',
                           class_weight='balanced', max_iter=1000)
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
    """Return a portfolio of classifiers.

    Returns
    -------
    classifiers : list of sklearn estimators
        List of classification models.
    """
    sklearn_ests = [
        LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=.1, solver='lbfgs', multi_class='multinomial'),
        RandomForestClassifier(max_features=None, n_estimators=100),
        RandomForestClassifier(max_features='sqrt', n_estimators=100),
        RandomForestClassifier(max_features='log2', n_estimators=100),
        SVC(C=1, gamma=0.03, kernel='rbf'),
        SVC(C=1, gamma='scale', kernel='rbf'),
        HistGradientBoostingClassifier(),

        HistGradientBoostingClassifier(
            l2_regularization=1e-08, learning_rate=0.01, loss='auto',
            max_bins=128, max_depth=19, max_iter=500, max_leaf_nodes=128,
            min_samples_leaf=38, n_iter_no_change=None, random_state=31537,
            scoring=None, tol=1e-07, validation_fraction=0.1, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=1e-10, learning_rate=0.1, loss='auto',
            max_bins=64, max_depth=2, max_iter=100, max_leaf_nodes=4,
            min_samples_leaf=3, n_iter_no_change=None, random_state=25689,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=1e-05, learning_rate=0.1, loss='auto',
            max_bins=255, max_depth=16, max_iter=400, max_leaf_nodes=64,
            min_samples_leaf=10, n_iter_no_change=None, random_state=58027,
            scoring=None, tol=1e-07, validation_fraction=0.1, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=10.0, learning_rate=0.1, loss='auto', max_bins=8,
            max_depth=20, max_iter=150, max_leaf_nodes=4, min_samples_leaf=13,
            n_iter_no_change=None, random_state=26894, scoring=None, tol=1e-07,
            validation_fraction=0.1, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=1e-07, learning_rate=0.01, loss='auto',
            max_bins=64, max_depth=15, max_iter=300, max_leaf_nodes=128,
            min_samples_leaf=8, n_iter_no_change=None, random_state=39911,
            scoring=None, tol=1e-07, validation_fraction=0.1, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=1e-08, learning_rate=0.1, loss='auto',
            max_bins=8, max_depth=6, max_iter=500, max_leaf_nodes=32,
            min_samples_leaf=15, n_iter_no_change=None, random_state=6477,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=10.0, learning_rate=0.1, loss='auto',
            max_bins=16, max_depth=4, max_iter=50, max_leaf_nodes=4,
            min_samples_leaf=6, n_iter_no_change=None, random_state=27348,
            scoring=None, tol=1e-07, validation_fraction=0.1, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=10.0, learning_rate=0.1, loss='auto',
            max_bins=255, max_depth=16, max_iter=100, max_leaf_nodes=128,
            min_samples_leaf=8, n_iter_no_change=None, random_state=10427,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=10.0, learning_rate=1.0, loss='auto',
            max_bins=255, max_depth=12, max_iter=250, max_leaf_nodes=32,
            min_samples_leaf=42, n_iter_no_change=None, random_state=2499,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=1e-08, learning_rate=0.1, loss='auto',
            max_bins=32, max_depth=6, max_iter=500, max_leaf_nodes=4,
            min_samples_leaf=19, n_iter_no_change=None, random_state=14210,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=0.1, learning_rate=0.01, loss='auto', max_bins=4,
            max_depth=18, max_iter=200, max_leaf_nodes=4, min_samples_leaf=39,
            n_iter_no_change=None, random_state=15428, scoring=None, tol=1e-07,
            validation_fraction=0.1, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=1e-06, learning_rate=0.1, loss='auto',
            max_bins=128, max_depth=12, max_iter=300, max_leaf_nodes=4,
            min_samples_leaf=3, n_iter_no_change=None, random_state=28019,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=0.0001, learning_rate=0.1, loss='auto',
            max_bins=128, max_depth=20, max_iter=500, max_leaf_nodes=128,
            min_samples_leaf=3, n_iter_no_change=None, random_state=22006,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=0.0001, learning_rate=0.1, loss='auto',
            max_bins=16, max_depth=7, max_iter=200, max_leaf_nodes=4,
            min_samples_leaf=4, n_iter_no_change=None, random_state=7320,
            scoring=None, tol=1e-07, validation_fraction=0.1, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=100.0, learning_rate=0.1, loss='auto',
            max_bins=255, max_depth=14, max_iter=500, max_leaf_nodes=16,
            min_samples_leaf=9, n_iter_no_change=None, random_state=15154,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=10.0, learning_rate=0.1, loss='auto',
            max_bins=255, max_depth=20, max_iter=400, max_leaf_nodes=64,
            min_samples_leaf=5, n_iter_no_change=None, random_state=18316,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=100000.0, learning_rate=1.0, loss='auto',
            max_bins=16, max_depth=2, max_iter=400, max_leaf_nodes=64,
            min_samples_leaf=14, n_iter_no_change=None, random_state=22357,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),

        HistGradientBoostingClassifier(
            l2_regularization=1e-05, learning_rate=0.1, loss='auto',
            max_bins=16, max_depth=None, max_iter=400, max_leaf_nodes=128,
            min_samples_leaf=48, n_iter_no_change=None, random_state=2136,
            scoring=None, tol=1e-07, validation_fraction=0.2, verbose=0),
        ]

    return sklearn_ests
