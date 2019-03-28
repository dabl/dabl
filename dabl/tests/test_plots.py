import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools

from sklearn.datasets import make_regression, make_blobs
from sklearn.preprocessing import KBinsDiscretizer
from dabl.preprocessing import clean, detect_types
from dabl.plotting import (plot_supervised, find_pretty_grid,
                           plot_classification_categorical,
                           plot_classification_continuous,
                           plot_regression_categorical,
                           plot_regression_continuous,
                           plot_coefficients)


# FIXME: check that target is not y but a column name


def test_find_pretty_grid():
    # test that the grid is big enough:
    rng = np.random.RandomState(0)
    for i in range(100):
        n_plots = rng.randint(1, 34)
        max_cols = rng.randint(1, 12)
        rows, cols = find_pretty_grid(n_plots=n_plots, max_cols=max_cols)
        assert rows * cols >= n_plots
        assert cols <= max_cols


@pytest.mark.parametrize("continuous_features, categorical_features, task",
                         itertools.product([0, 1, 3, 100], [0, 1, 3, 100],
                                           ['classification', 'regression']))
def test_plots_smoke(continuous_features, categorical_features, task):
    # simple smoke test
    # should be parametrized
    n_samples = 100
    X_cont, y_cont = make_regression(
        n_samples=n_samples, n_features=continuous_features,
        n_informative=min(continuous_features, 2))
    X_cat, y_cat = make_regression(
        n_samples=n_samples, n_features=categorical_features,
        n_informative=min(categorical_features, 2))
    if X_cat.shape[1] > 0:
        X_cat = KBinsDiscretizer(encode='ordinal').fit_transform(X_cat)
    cont_columns = ["asdf_%d_cont" % i for i in range(continuous_features)]
    df_cont = pd.DataFrame(X_cont, columns=cont_columns)
    if categorical_features > 0:
        cat_columns = ["asdf_%d_cat" % i for i in range(categorical_features)]
        df_cat = pd.DataFrame(X_cat, columns=cat_columns).astype('int')
        df_cat = df_cat.astype("category")
        X_df = pd.concat([df_cont, df_cat], axis=1)
    else:
        X_df = df_cont
    assert(X_df.shape[1] == continuous_features + categorical_features)
    X_clean = clean(X_df.copy())
    y = y_cont + y_cat
    if X_df.shape[1] == 0:
        y = np.random.uniform(size=n_samples)
    if task == "classification":
        y = np.digitize(y, np.percentile(y, [5, 10, 60, 85]))
    X_clean['target'] = y
    if task == "classification":
        X_clean['target'] = X_clean['target'].astype('category')
    types = detect_types(X_clean)
    column_types = types.T.idxmax()
    assert np.all(column_types[:continuous_features] == 'continuous')
    assert np.all(column_types[continuous_features:-1] == 'categorical')
    if task == "classification":
        assert column_types[-1] == 'categorical'
    else:
        assert column_types[-1] == 'continuous'

    plot_supervised(X_clean, 'target')
    plt.close("all")


@pytest.mark.filterwarnings('ignore:Discarding near-constant')
def test_plot_classification_n_classes():
    X, y = make_blobs()
    X = pd.DataFrame(X)
    X['target'] = 0
    with pytest.raises(ValueError, match="Less than two classes"):
        plot_classification_categorical(X, 'target')
    with pytest.raises(ValueError, match="Less than two classes"):
        plot_classification_continuous(X, 'target')


def test_plot_wrong_target_type():
    X, y = make_blobs()
    X = pd.DataFrame(X)
    X['target'] = y
    with pytest.raises(ValueError, match="need continuous"):
        plot_regression_categorical(X, 'target')
    with pytest.raises(ValueError, match="need continuous"):
        plot_regression_continuous(X, 'target')

    X['target'] = X[0]
    with pytest.raises(ValueError, match="need categorical"):
        plot_classification_categorical(X, 'target')
    with pytest.raises(ValueError, match="need categorical"):
        plot_classification_continuous(X, 'target')


@pytest.mark.parametrize("n_features, n_top_features",
                         [(5, 10), (10, 5), (10, 40), (40, 10)])
def test_plot_coefficients(n_features, n_top_features):
    coef = np.arange(n_features) + .4
    names = ["feature_{}".format(i) for i in range(n_features)]
    plot_coefficients(coef, names, n_top_features=n_top_features)
    ax = plt.gca()
    assert len(ax.get_xticks()) == min(n_top_features, n_features)
    coef[:-5] = 0
    plot_coefficients(coef, names, n_top_features=n_top_features)
    ax = plt.gca()
    assert len(ax.get_xticks()) == 5

