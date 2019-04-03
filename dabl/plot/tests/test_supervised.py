import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools

from sklearn.datasets import make_regression, make_blobs
from sklearn.preprocessing import KBinsDiscretizer
from dabl.preprocessing import clean, detect_types
from dabl.plot.supervised import (
    plot_supervised, plot_classification_categorical,
    plot_classification_continuous, plot_regression_categorical,
    plot_regression_continuous)


# FIXME: check that target is not y but a column name

@pytest.mark.filterwarnings('ignore:the matrix subclass')
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


@pytest.mark.parametrize("add, feature_type, target_type",
                         itertools.product([0, .1],
                                           ['continuous', 'categorical'],
                                           ['continuous', 'categorical']))
def test_type_hints(add, feature_type, target_type):
    X = pd.DataFrame(np.random.randint(4, size=100)) + add
    X['target'] = np.random.uniform(size=100)
    plot_supervised(X, type_hints={0: feature_type,
                                   'target': target_type},
                    target_col='target')
    # get title of figure
    text = plt.gcf()._suptitle.get_text()
    assert feature_type.capitalize() in text


def test_float_classification_target():
    # check we can plot even if we do classification with a float target
    X, y = make_blobs()
    data = pd.DataFrame(X)
    data['target'] = y.astype(np.float)
    types = detect_types(data)
    assert types.categorical['target']
    plot_supervised(data, 'target')
    # same with "actual float" - we need to specify classification for that :-/
    data['target'] = y.astype(np.float) + .2
    plot_supervised(data, 'target', type_hints={'target': 'categorical'})
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