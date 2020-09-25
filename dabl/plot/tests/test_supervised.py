import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools

from sklearn.datasets import (make_regression, make_blobs, load_digits,
                              fetch_openml, load_diabetes)
from sklearn.preprocessing import KBinsDiscretizer
from dabl.preprocessing import clean, detect_types, guess_ordinal
from dabl.plot.supervised import (
    plot, plot_classification_categorical,
    plot_classification_continuous, plot_regression_categorical,
    plot_regression_continuous, _get_scatter_alpha, _get_scatter_size)
from dabl.utils import data_df_from_bunch


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

    plot(X_clean, target_col='target')
    plt.close("all")


@pytest.mark.parametrize("add, feature_type, target_type",
                         itertools.product([0, .1],
                                           ['continuous', 'categorical'],
                                           ['continuous', 'categorical']))
def test_type_hints(add, feature_type, target_type):
    X = pd.DataFrame(np.random.randint(4, size=100)) + add
    X['target'] = np.random.uniform(size=100)
    plot(X, type_hints={0: feature_type,
                        'target': target_type},
         target_col='target')
    # get title of figure
    text = plt.gcf()._suptitle.get_text()
    assert feature_type.capitalize() in text
    ax = plt.gca()
    # one of the labels is 'target' iif regression
    labels = ax.get_ylabel() + ax.get_xlabel()
    assert ('target' in labels) == (target_type == 'continuous')
    plt.close("all")


def test_float_classification_target():
    # check we can plot even if we do classification with a float target
    X, y = make_blobs()
    data = pd.DataFrame(X)
    data['target'] = y.astype(np.float)
    types = detect_types(data)
    assert types.categorical['target']
    plot(data, target_col='target')
    # same with "actual float" - we need to specify classification for that :-/
    data['target'] = y.astype(np.float) + .2
    plot(data, target_col='target', type_hints={'target': 'categorical'})
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


def test_plot_target_low_card_int():
    data = load_digits()
    df = data_df_from_bunch(data)
    plot(df[::10], target_col='target')


def test_plot_X_y():
    X, y = make_blobs()
    X = pd.DataFrame(X)
    plot(X, y)


def test_plot_regression_numpy():
    X, y = make_regression()
    plot(X, y)


def test_plot_lda_binary():
    X, y = make_blobs(centers=2)
    X = pd.DataFrame(X)
    plot(X, y, univariate_plot='kde')


def test_plot_int_column_name():
    X, y = make_blobs()
    X = pd.DataFrame(X)
    X[3] = y
    plot(X, target_col=3)


def test_negative_ordinal():
    # check that a low card int with negative values is plotted correctly
    data = pd.DataFrame([np.random.randint(0, 10, size=1000) - 5,
                         np.random.randint(0, 2, size=1000)]).T
    # ensure first column is low_card_int
    assert (detect_types(data).T.idxmax()
            == ['low_card_int', 'categorical']).all()
    assert guess_ordinal(data[0])
    # smoke test
    plot(data, target_col=1)


def test_plot_classification_continuous():
    data = fetch_openml('MiceProtein')
    df = data_df_from_bunch(data)
    # only univariate plots
    figures = plot_classification_continuous(df, target_col='target',
                                             plot_pairwise=False)
    assert len(figures) == 1
    # top 10 axes
    assert len(figures[0].get_axes()) == 10
    # six is the minimum number of features for histograms
    # (last column is target)
    figures = plot_classification_continuous(df.iloc[:, -7:],
                                             target_col='target',
                                             plot_pairwise=False)
    assert len(figures) == 1
    assert len(figures[0].get_axes()) == 6

    # for 5 features, do full pairplot
    figures = plot_classification_continuous(df.iloc[:, -6:],
                                             target_col='target',
                                             plot_pairwise=False)
    assert len(figures) == 1
    # diagonal has twin axes
    assert len(figures[0].get_axes()) == 5 * 5 + 5

    # also do pairwise plots
    figures = plot_classification_continuous(df, target_col='target',
                                             random_state=42)
    # univariate, pairwise, pca, lda
    assert len(figures) == 4
    # univariate
    axes = figures[0].get_axes()
    assert len(axes) == 10
    # known result
    assert axes[0].get_xlabel() == "SOD1_N"
    # bar plot never has ylabel
    assert axes[0].get_ylabel() == ""
    # pairwise
    axes = figures[1].get_axes()
    assert len(axes) == 4
    # known result
    assert axes[0].get_xlabel() == "SOD1_N"
    assert axes[0].get_ylabel() == 'S6_N'

    # PCA
    axes = figures[2].get_axes()
    assert len(axes) == 4
    # known result
    assert axes[0].get_xlabel() == "PCA 1"
    assert axes[0].get_ylabel() == 'PCA 5'

    # LDA
    axes = figures[3].get_axes()
    assert len(axes) == 4
    # known result
    assert axes[0].get_xlabel() == "LDA 0"
    assert axes[0].get_ylabel() == 'LDA 1'


def test_plot_string_target():
    X, y = make_blobs(n_samples=30)
    data = pd.DataFrame(X)
    y = pd.Series(y)
    y[y == 0] = 'a'
    y[y == 1] = 'b'
    y[y == 2] = 'c'
    data['target'] = y
    plot(data, target_col='target')


def test_na_vals_reg_plot_raise_warning():
    X, y = load_diabetes(return_X_y=True)
    X = pd.DataFrame(X)
    y[::50] = np.NaN
    X['target_col'] = y
    scatter_alpha = _get_scatter_alpha('auto', X['target_col'])
    scatter_size = _get_scatter_size('auto', X['target_col'])
    with pytest.warns(UserWarning, match="Missing values in target_col have "
                                         "been removed for regression"):
        plot(X, 'target_col')
    with pytest.warns(UserWarning, match="Missing values in target_col have "
                                         "been removed for regression"):
        plot_regression_continuous(X, 'target_col',
                                   scatter_alpha=scatter_alpha,
                                   scatter_size=scatter_size)
    with pytest.warns(UserWarning, match="Missing values in target_col have "
                                         "been removed for regression"):
        plot_regression_categorical(X, 'target_col',
                                    scatter_alpha=scatter_alpha,
                                    scatter_size=scatter_size)


def test_plot_regression_continuous_with_target_outliers():
    df = pd.DataFrame(
        data={
            "feature": np.random.randint(low=1, high=100, size=200),
            # target values are bound between 50 and 100
            "target": np.random.randint(low=50, high=100, size=200)
            }
    )
    scatter_alpha = _get_scatter_alpha('auto', df['target'])
    scatter_size = _get_scatter_size('auto', df['target'])
    # append single outlier record with target value 0
    df = df.append({"feature": 50, "target": 0}, ignore_index=True)

    with pytest.warns(
        UserWarning,
        match="Dropped 1 outliers in column target."
    ):
        plot_regression_continuous(df, 'target',
                                   scatter_alpha=scatter_alpha,
                                   scatter_size=scatter_size
                                   )
