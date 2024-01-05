import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import itertools

from sklearn.datasets import (make_regression, make_blobs, load_digits,
                              fetch_openml, load_diabetes)
from sklearn.preprocessing import KBinsDiscretizer
from dabl.preprocessing import clean, detect_types, guess_ordinal
from dabl.plot.supervised import (
    plot, plot_classification_categorical,
    plot_classification_continuous, plot_regression_categorical,
    plot_regression_continuous)
from dabl.utils import data_df_from_bunch
from dabl import set_config


# FIXME: check that target is not y but a column name

@pytest.mark.filterwarnings('ignore:the matrix subclass')
@pytest.mark.parametrize("continuous_features, categorical_features, task",
                         itertools.product([0, 1, 3, 100], [0, 1, 3, 100],
                                           ['classification', 'regression']))
def test_plots_smoke(continuous_features, categorical_features, task):
    # simple smoke test
    # should be parametrized
    if continuous_features == 0 and categorical_features == 0:
        pytest.skip("Need at least one feature")
    n_samples = 100
    if continuous_features > 0:
        X_cont, y_cont = make_regression(
            n_samples=n_samples, n_features=continuous_features,
            n_informative=min(continuous_features, 2))
    if categorical_features > 0:
        X_cat, y_cat = make_regression(
            n_samples=n_samples, n_features=categorical_features,
            n_informative=min(categorical_features, 2))
    if continuous_features > 0:
        cont_columns = ["asdf_%d_cont" % i for i in range(continuous_features)]
        df_cont = pd.DataFrame(X_cont, columns=cont_columns)
    if categorical_features > 0:
        X_cat = KBinsDiscretizer(encode='ordinal').fit_transform(X_cat)
        cat_columns = ["asdf_%d_cat" % i for i in range(categorical_features)]
        df_cat = pd.DataFrame(X_cat, columns=cat_columns).astype('int')
        df_cat = df_cat.astype("category")
    if categorical_features > 0 and continuous_features > 0:
        X_df = pd.concat([df_cont, df_cat], axis=1)
        y = y_cont + y_cat
    elif categorical_features > 0:
        X_df = df_cat
        y = y_cat
    elif continuous_features > 0:
        X_df = df_cont
        y = y_cont
    else:
        raise ValueError("invalid")
    assert X_df.shape[1] == continuous_features + categorical_features
    X_clean = clean(X_df.copy())
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
    data['target'] = y.astype(float)
    types = detect_types(data)
    assert types.categorical['target']
    plot(data, target_col='target')
    # same with "actual float" - we need to specify classification for that :-/
    data['target'] = y.astype(float) + .2
    plot(data, target_col='target', type_hints={'target': 'categorical'})
    plt.close("all")


@pytest.mark.filterwarnings('ignore:Discarding near-constant')
def test_plot_classification_n_classes():
    X, y = make_blobs()
    X = pd.DataFrame(X)
    X['target'] = 0
    with pytest.raises(ValueError, match="Less than two classes"):
        plot_classification_categorical(X, target_col='target')
    with pytest.raises(ValueError, match="Less than two classes"):
        plot_classification_continuous(X, target_col='target')


def test_plot_wrong_target_type():
    X, y = make_blobs()
    X = pd.DataFrame(X)
    X['target'] = y
    with pytest.raises(ValueError, match="need continuous"):
        plot_regression_categorical(X, target_col='target')
    with pytest.raises(ValueError, match="need continuous"):
        plot_regression_continuous(X, target_col='target')

    X['target'] = X[0]
    with pytest.raises(ValueError, match="need categorical"):
        plot_classification_categorical(X, target_col='target')
    with pytest.raises(ValueError, match="need categorical"):
        plot_classification_continuous(X, target_col='target')


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
    # FIXME this is a weird example for ordinal, as it's uniform?
    assert (detect_types(data).T.idxmax()
            == ['low_card_int_ordinal', 'categorical']).all()
    # smoke test
    plot(data, target_col=1)


def test_large_ordinal():
    # check that large integers don't bring us down (bincount memory error)
    # here some random phone numbers
    assert not guess_ordinal(pd.Series([6786930208, 2142878625, 9106275431]))


def test_detect_low_cardinality_int():
    rng = np.random.RandomState(42)
    df_all = pd.DataFrame(
        {'binary_int': rng.randint(0, 2, size=1000),
         'categorical_int': rng.randint(0, 4, size=1000),
         'low_card_int_uniform': rng.randint(0, 20, size=1000),
         'low_card_int_binomial': rng.binomial(20, .3, size=1000),
         'cont_int': np.repeat(np.arange(500), 2),
         })

    res = detect_types(df_all)
    types = res.T.idxmax()
    # This is duplicated from a preprocessing test, but let's make sure this behavior is as expected
    assert types['binary_int'] == 'categorical'
    assert types['categorical_int'] == 'categorical'
    assert types['low_card_int_uniform'] == 'low_card_int_categorical'
    assert types['low_card_int_binomial'] == 'low_card_int_ordinal'
    assert types['cont_int'] == 'continuous'
    classification_plots = plot(df_all, target_col="binary_int", plot_pairwise=False)
    assert len(classification_plots) == 3
    # scatter matrix of two continuous features
    assert classification_plots[1][0][0, 0].get_ylabel() == "low_card_int_binomial"
    assert classification_plots[1][0][1, 0].get_ylabel() == "cont_int"
    assert classification_plots[2].shape == (1, 2)
    assert classification_plots[2][0, 0].get_title() == "low_card_int_uniform"
    assert classification_plots[2][0, 1].get_title() == "categorical_int"

    regression_plots = plot(df_all, target_col="cont_int")
    assert len(regression_plots) == 3
    assert regression_plots[1].shape == (1, 1)
    assert regression_plots[1][0, 0].get_xlabel() == "low_card_int_binomial (jittered)"
    assert regression_plots[2].shape == (1, 3)
    assert regression_plots[2][0, 0].get_ylabel() == "binary_int"
    assert regression_plots[2][0, 1].get_ylabel() == "categorical_int"
    assert regression_plots[2][0, 2].get_ylabel() == "low_card_int_uniform"


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
    assert figures[0].size == 5 * 5

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
    axes = figures[1].ravel()
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


def test_plot_classification_many_classes():
    X, y = make_blobs(n_samples=200)
    data = pd.DataFrame(X)
    y = pd.Series(y)
    y[:20] = np.arange(20)
    data['target'] = y
    plot(data, target_col='target', type_hints={'target': 'categorical'})


def test_plot_string_target():
    X, y = make_blobs(n_samples=30)
    data = pd.DataFrame(X)
    y = pd.Series(y)
    y[y == 0] = 'a'
    y[y == 1] = 'b'
    y[y == 2] = 'c'
    data['target'] = y
    plot(data, target_col='target')


def test_plot_mixed_column_name_types():
    X, y = make_blobs(n_samples=100, n_features=10)
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
    with pytest.warns(UserWarning, match="Missing values in target_col have "
                                         "been removed for regression"):
        plot(X, 'target_col')
    with pytest.warns(UserWarning, match="Missing values in target_col have "
                                         "been removed for regression"):
        plot_regression_continuous(X, target_col='target_col')
    with pytest.warns(UserWarning, match="Missing values in target_col have "
                                         "been removed for regression"):
        plot_regression_categorical(X, target_col='target_col')


def test_plot_regression_with_target_outliers():
    df = pd.DataFrame(
        data={
            "feature": np.random.randint(low=1, high=100, size=200),
            # target values are bound between 50 and 100
            "target": np.random.randint(low=50, high=100, size=200)
            }
    )
    # append single outlier record with target value 0
    df = pd.concat([df, pd.DataFrame({"feature": [50], "target": [0]})], ignore_index=True)

    with pytest.warns(
        UserWarning,
        match="Dropped 1 outliers in column target."
    ):
        plot_regression_continuous(df, target_col='target')

    with pytest.warns(
        UserWarning,
        match="Dropped 1 outliers in column target."
    ):
        plot_regression_categorical(df, target_col='target')

    res = plot(df, target_col='target')
    assert len(res) == 3
    ax = res[0]
    # ensure outlier at 0 was removed
    assert ax.get_xticks()[0] == 40


def test_plot_regression_categorical_missing_value():
    df = pd.DataFrame({'y': np.random.normal(size=300)})
    df.loc[100:200, 'y'] += 1
    df.loc[200:300, 'y'] += 2
    df['x'] = 'a'
    df.loc[100:200, 'x'] = 'b'
    df.loc[200:300, 'x'] = np.NaN
    res = plot(df, target_col='y')
    assert len(res[2][0, 0].get_yticklabels()) == 3
    assert res[2][0, 0].get_yticklabels()[2].get_text() == 'dabl_mi...'


def test_plot_regression_missing_categories():
    df = pd.DataFrame({'cat_col': np.random.choice(['a', 'b', 'c', 'd'],
                                                   size=100)})
    df['target'] = np.NaN
    counts = df.cat_col.value_counts()
    df.loc[df.cat_col == "a", 'target'] = np.random.normal(size=counts['a'])
    df.loc[df.cat_col == "b", 'target'] = np.random.normal(1, size=counts['b'])
    axes = plot(df, target_col="target")
    ticklabels = axes[-1][0, 0].get_yticklabels()
    assert [label.get_text() for label in ticklabels] == ['a', 'b']


def test_plot_regression_correlation():
    df = pd.DataFrame({'y': np.random.normal(size=1000)})
    df['x1'] = df['y'] + np.random.normal(scale=.1, size=1000)
    df['x2'] = df['x1'] + np.random.normal(scale=.1, size=1000)
    with pytest.warns(
        UserWarning,
        match=r"Not plotting highly correlated \(0.*\) feature x2"
    ):
        res = plot_regression_continuous(df, target_col="y")
    assert res.shape == (1, 1)
    with warnings.catch_warnings(record=True) as w:
        res = plot_regression_continuous(
            df, target_col="y", prune_correlations_threshold=0)
        assert len(w) == 0
    assert res.shape == (1, 2)


def test_plot_regression_categoricals_scatter():
    rng = np.random.RandomState(0)
    data = pd.DataFrame(rng.normal(scale=4, size=(1000, 2)),
                        columns=["cont1", "cont2"])
    data['cat1'] = 1 - 2 * rng.randint(0, 2, size=1000)
    data['cat2'] = 1 - 2 * rng.randint(0, 2, size=1000)
    data['y'] = (data.cat1 * (data.cont1 + 2) ** 2 - 10 * data.cat1
                 + data.cont1 * 0.5 + data.cat2 * data.cont2 * 3)
    figs = plot(data, target_col="y", find_scatter_categoricals=True)
    ax1, ax2 = figs[1][0]
    assert ax1.get_xlabel() == 'cont2'
    assert ax2.get_xlabel() == 'cont1'
    assert ax1.get_legend().get_title().get_text() == "cat2"
    assert ax2.get_legend().get_title().get_text() == "cat1"


def test_label_truncation():
    a = ('a_really_long_name_that_would_mess_up_the_layout_a_lot'
         '_by_just_being_very_long')
    b = ('the_target_that_has_an_equally_long_name_which_would_'
         'mess_up_everything_as_well_but_in_different_places')
    df = pd.DataFrame({a: np.random.uniform(0, 1, 1000)})
    df[b] = df[a] + np.random.uniform(0, 0.1, 1000)
    res = plot_regression_continuous(df, target_col=b)

    assert res[0, 0].get_ylabel() == 'the_target_that_h...'
    assert (res[0, 0].get_xlabel()
            == 'a_really_long_name_that_would_mess_up_the_layou...')

    set_config(truncate_labels=False)
    res = plot_regression_continuous(df, target_col=b)

    assert res[0, 0].get_ylabel() == b
    assert res[0, 0].get_xlabel() == a
    set_config(truncate_labels=True)
