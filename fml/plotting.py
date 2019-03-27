import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from warnings import warn


from sklearn.feature_selection import (f_regression,
                                       mutual_info_regression,
                                       mutual_info_classif,
                                       f_classif)
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.graphics.mosaicplot import mosaic

from .preprocessing import detect_types


def find_pretty_grid(n_plots, max_cols=5):
    """Determine a good grid shape for n_plots subplots.

    Tries to find a way to arange n_plots many subplots on a grid in a way
    that fills as many grid-cells as possible, while keeping the number
    of rows low and the number of columns below max_cols.

    Parameters
    ----------
    n_plots : int
        Number of plots to arrange.
    max_cols : int, default=5
        Maximum number of columns.

    Returns
    -------
    n_rows : int
        Number of rows in grid.
    n_cols : int
        Number of columns in grid.

    Examples
    --------
    >>> find_pretty_grid(16, 5)
    (4, 4)
    >>> find_pretty_grid(11, 5)
    (3, 4)
    >>> find_pretty_grid(10, 5)
    (2, 5)
    """

    # we could probably do something with prime numbers here
    # but looks like that becomes a combinatorial problem again?
    if n_plots % max_cols == 0:
        # perfect fit!
        # if max_cols is 6 do we prefer 6x1 over 3x2?
        return int(n_plots / max_cols), max_cols
    # min number of rows needed
    min_rows = int(np.ceil(n_plots / max_cols))
    best_empty = max_cols
    best_cols = max_cols
    for cols in range(max_cols, min_rows - 1, -1):
        # we only allow getting narrower if we have more cols than rows
        remainder = (n_plots % cols)
        empty = cols - remainder if remainder != 0 else 0
        if empty == 0:
            return int(n_plots / cols), cols
        if empty < best_empty:
            best_empty = empty
            best_cols = cols
    return int(np.ceil(n_plots / best_cols)), best_cols


def plot_continuous_unsupervised(X):
    """Not implemented yet"""
    pass


def plot_categorical_unsupervised(X):
    """not implemented yet"""
    pass


def _shortname(some_string, maxlen=20):
    """Shorten a string given a maximum length.

    Longer strings will be shortened and the rest replaced by ...

    Parameters
    ----------
    some_string : string
        Input string to shorten
    maxlen : int, default=20

    Returns
    -------
    return_string : string
        Output string of size ``min(len(some_string), maxlen)``.
    """
    some_string = str(some_string)
    if len(some_string) > maxlen:
        return some_string[:maxlen - 3] + "..."
    else:
        return some_string


def _get_n_top(features, name):
    if features.shape[1] > 20:
        print("Showing only top 10 of {} {} features".format(
            features.shape[1], name))
        # too many features, show just top 10
        show_top = 10
    else:
        show_top = features.shape[1]
    return show_top


def _prune_categories(series, max_categories=10):
    series = series.astype('category')
    small_categories = series.value_counts()[max_categories:].index
    res = series.cat.remove_categories(small_categories)
    res = res.cat.add_categories(['fml_other']).fillna("fml_other")
    return res


def _prune_category_make_X(X, col, target_col):
    col_values = X[col]
    if col_values.nunique() > 20:
        # keep only top 10 categories if there are more than 20
        col_values = _prune_categories(col_values)
        X_new = X[[target_col]].copy()
        X_new[col] = col_values
    else:
        X_new = X.copy()
        X_new[col] = X_new[col].astype('category')
    return X_new


def _fill_missing_categorical(X):
    # fill in missing values in categorical variables with new category
    # ensure we use strings for object columns and number for integers
    X = X.copy()
    max_value = X.max(numeric_only=True).max()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna("fml_missing", inplace=True)
        else:
            X[col].fillna(max_value + 1, inplace=True)
    return X


def plot_unsupervised(X, verbose=10):
    """Not implemented yet"""
    types = detect_types(X)
    # if any dirty floats, tell user to clean them first
    plot_continuous_unsupervised(X.loc[:, types.continous])
    plot_categorical_unsupervised(X.loc[:, types.categorical])


def plot_regression_continuous(X, target_col, types=None):
    """Exploration plots for continuous features in regression.

    Creates plots of all the continuous features vs the target.
    Relevant features are determined using F statistics.

    Parameters
    ----------
    X : dataframe
        Input data including features and target
    target_col : str or int
        Identifier of the target column in X
    types : dataframe of types, optional.
        Output of detect_types on X. Can be used to avoid recomputing the
        types.
    """
    if types is None:
        types = detect_types(X)
    features = X.loc[:, types.continuous]
    if target_col in features.columns:
        features = features.drop(target_col, axis=1)
    if features.shape[1] == 0:
        return
    show_top = _get_n_top(features, "continuous")

    target = X[target_col]
    # HACK we should drop them per column before feeding them into f_regression
    # FIXME
    features_imp = SimpleImputer().fit_transform(features)
    f, p = f_regression(features_imp, target)
    top_k = np.argsort(f)[-show_top:][::-1]
    # we could do better lol
    fig, axes = _make_subplots(n_plots=show_top)

    # FIXME this could be a function or maybe using seaborn
    plt.suptitle("Continuous Feature vs Target")
    for i, (col, ax) in enumerate(zip(top_k, axes.ravel())):
        if i % axes.shape[1] == 0:
            ax.set_ylabel(target_col)
        ax.plot(features.iloc[:, col], target, 'o', alpha=.6)
        ax.set_xlabel(_shortname(features.columns[col]))
        ax.set_title("F={:.2E}".format(f[col]))

    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()


def _make_subplots(n_plots, max_cols=5, row_height=3):
    n_rows, n_cols = find_pretty_grid(n_plots, max_cols=max_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, row_height * n_rows),
                             constrained_layout=True)
    # we don't want ravel to fail, this is awkward!
    axes = np.atleast_2d(axes)
    return fig, axes


def plot_regression_categorical(X, target_col, types=None):
    """Exploration plots for categorical features in regression.

    Creates box plots of target distribution for important categorical
    features. Relevant features are identified using mutual information.

    For high cardinality categorical variables (variables with many categories)
    only the most frequent categories are shown.

    Parameters
    ----------
    X : dataframe
        Input data including features and target
    target_col : str or int
        Identifier of the target column in X
    types : dataframe of types, optional.
        Output of detect_types on X. Can be used to avoid recomputing the
        types.
    """
    if types is None:
        types = detect_types(X)
    features = X.loc[:, types.categorical]
    if target_col in features.columns:
        features = features.drop(target_col, axis=1)
    if features.shape[1] == 0:
        return
    features = features.astype('category')
    show_top = _get_n_top(features, "categorical")
    # for col in X.columns:
    #    if col != target_col:
    #        X[col] = X[col].astype("category")
    # seaborn needs to know these are categories
    # can't use OrdinalEncoder because we might have mix of int and string
    ordinal_encoded = features.apply(lambda x: x.cat.codes)
    target = X[target_col]
    f = mutual_info_regression(
        ordinal_encoded, target,
        discrete_features=np.ones(X.shape[1], dtype=bool))
    top_k = np.argsort(f)[-show_top:][::-1]

    # large number of categories -> taller plot
    row_height = 3 if X.nunique().max() <= 5 else 5
    fig, axes = _make_subplots(n_plots=show_top, row_height=row_height)
    plt.suptitle("Categorical Feature vs Target")
    for i, (col_ind, ax) in enumerate(zip(top_k, axes.ravel())):
        col = features.columns[i]
        X_new = _prune_category_make_X(X, col, target_col)
        medians = X_new.groupby(col)[target_col].median()
        order = medians.sort_values().index
        sns.boxplot(x=target_col, y=col, data=X_new, order=order, ax=ax)
        ax.set_title("F={:.2E}".format(f[col_ind]))
        # shorten long ticks and labels
        _short_tick_names(ax)

    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()


def _short_tick_names(ax):
    ax.set_yticklabels([_shortname(t.get_text(), maxlen=10)
                        for t in ax.get_yticklabels()])
    ax.set_xlabel(_shortname(ax.get_xlabel(), maxlen=20))
    ax.set_ylabel(_shortname(ax.get_ylabel(), maxlen=20))


def _find_scatter_plots_classification(X, target):
    # input is continuous
    # look at all pairs of features, find most promising ones
    dummy = DummyClassifier(strategy='prior').fit(X, target)
    baseline_score = recall_score(target, dummy.predict(X), average='macro')
    scores = []
    for i, j in itertools.combinations(np.arange(X.shape[1]), 2):
        this_X = X[:, [i, j]]
        # assume this tree is simple enough so not be able to overfit in 2d
        # so we don't bother with train/test split
        tree = DecisionTreeClassifier(max_leaf_nodes=8).fit(this_X, target)
        scores.append((i, j, np.mean(cross_val_score(
            tree, this_X, target, cv=5, scoring='recall_macro'))))
        # scores.append((i, j, recall_score(target, tree.predict(this_X),
        #                                  average='macro')))
    scores = pd.DataFrame(scores, columns=['feature0', 'feature1', 'score'])
    top_3 = scores.sort_values(by='score').iloc[-3:][::-1]
    print("baseline score: {:.3f}".format(baseline_score))
    return top_3


def _discrete_scatter(x, y, c, ax):
    for i in np.unique(c):
        mask = c == i
        ax.plot(x[mask], y[mask], 'o', label=i)
    ax.legend()


def plot_classification_continuous(X, target_col, types=None):
    """Exploration plots for continuous features in classification.

    Selects important continuous features according to F statistics.
    Creates univariate distribution plots for these, as well as scatterplots
    for selected pairs of features, and scatterplots for selected pairs of
    PCA directions.
    If there are more than 2 classes, scatter plots from Linear Discriminant
    Analysis are also shown.
    Scatter plots are determined "interesting" is a decision tree on the
    two-dimensional projection performs well. The cross-validated macro-average
    recall of a decision tree is shown in the title for each scatterplot.

    Parameters
    ----------
    X : dataframe
        Input data including features and target
    target_col : str or int
        Identifier of the target column in X
    types : dataframe of types, optional.
        Output of detect_types on X. Can be used to avoid recomputing the
        types.
    """
    if types is None:
        types = detect_types(X)
    features = X.loc[:, types.continuous]
    if target_col in features.columns:
        features = features.drop(target_col, axis=1)
    if features.shape[1] == 0:
        return
    top_for_interactions = 20
    features_imp = SimpleImputer().fit_transform(features)
    target = X[target_col]
    # FIXME if one class only has NaN for a value we crash! :-/
    # TODO univariate plot?
    # already on diagonal for pairplot but not for many features
    if features.shape[1] <= 5:
        # for n_dim <= 5 we do full pairplot plot
        # FIXME filling in missing values here b/c of a bug in seaborn
        # we really shouldn't be doing this
        # https://github.com/mwaskom/seaborn/issues/1699
        X_imp = X.fillna(features.median(axis=0))
        sns.pairplot(X_imp, vars=features.columns,
                     hue=target_col)
    else:
        # univariate plots
        show_top = _get_n_top(features, "continuous")
        f, p = f_classif(features_imp, target)
        top_k = np.argsort(f)[-show_top:][::-1]
        # FIXME this will fail if a feature is always
        # NaN for a particular class
        best_features = features.iloc[:, top_k].copy()

        best_features[target_col] = target
        df = best_features.melt(target_col)
        rows, cols = find_pretty_grid(show_top)
        g = sns.FacetGrid(df, col='variable', hue=target_col, col_wrap=cols,
                          sharey=False, sharex=False)
        g = g.map(sns.kdeplot, "value", shade=True)
        # FIXME remove "variable = " from title, add f score
        plt.suptitle("Univariate Distributions", y=1.02)

        # pairwise plots
        top_k = np.argsort(f)[-top_for_interactions:][::-1]
        top_pairs = _find_scatter_plots_classification(
            features_imp[:, top_k], target)
        fig, axes = plt.subplots(1, len(top_pairs),
                                 figsize=(len(top_pairs) * 4, 4))
        for x, y, score, ax in zip(top_pairs.feature0, top_pairs.feature1,
                                   top_pairs.score, axes.ravel()):
            i = top_k[x]
            j = top_k[y]
            _discrete_scatter(features_imp[:, i], features_imp[:, j],
                              c=target, ax=ax)
            ax.set_xlabel(features.columns[i])
            ax.set_ylabel(features.columns[j])
            ax.set_title("{:.3f}".format(score))
        fig.suptitle("Top feature interactions")
    # get some PCA directions
    # we're using all features here, not only most informative
    # should we use only those?
    n_components = min(top_for_interactions, features.shape[0],
                       features.shape[1])

    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(scale(features_imp))
    top_pairs = _find_scatter_plots_classification(features_pca, target)
    # copy and paste from above. Refactor?
    fig, axes = plt.subplots(1, len(top_pairs),
                             figsize=(len(top_pairs) * 4, 4))
    if len(top_pairs) <= 1:
        # we don't want ravel to fail, this is awkward!
        axes = np.array([axes])
    for x, y, score, ax in zip(top_pairs.feature0, top_pairs.feature1,
                               top_pairs.score, axes.ravel()):

        _discrete_scatter(features_pca[:, x], features_pca[:, y],
                          c=target, ax=ax)
        ax.set_xlabel("PCA {}".format(x))
        ax.set_ylabel("PCA {}".format(y))
        ax.set_title("{:.3f}".format(score))
    fig.suptitle("Discriminating PCA directions")
    # LDA
    lda = LinearDiscriminantAnalysis(
        n_components=min(n_components, target.nunique() - 1))
    features_lda = lda.fit_transform(scale(features_imp), target)
    top_pairs = _find_scatter_plots_classification(features_lda, target)
    # copy and paste from above. Refactor?
    fig, axes = plt.subplots(1, len(top_pairs),
                             figsize=(len(top_pairs) * 4, 4))
    if len(top_pairs) <= 1:
        # we don't want ravel to fail, this is awkward!
        axes = np.array([axes])
    for x, y, score, ax in zip(top_pairs.feature0, top_pairs.feature1,
                               top_pairs.score, axes.ravel()):

        _discrete_scatter(features_pca[:, x], features_pca[:, y],
                          c=target, ax=ax)
        ax.set_xlabel("LDA {}".format(x))
        ax.set_ylabel("LDA {}".format(y))
        ax.set_title("{:.3f}".format(score))
    fig.suptitle("Discriminating LDA directions")
    # TODO fancy manifolds?


def plot_classification_categorical(X, target_col, types=None, kind='count'):
    """Exploration plots for categorical features in classification.

    Creates plots of categorical variable distributions for each target class.
    Relevant features are identified via mutual information.

    For high cardinality categorical variables (variables with many categories)
    only the most frequent categories are shown.

    Parameters
    ----------
    X : dataframe
        Input data including features and target
    target_col : str or int
        Identifier of the target column in X
    types : dataframe of types, optional.
        Output of detect_types on X. Can be used to avoid recomputing the
        types.
    """
    if types is None:
        types = detect_types(X)
    features = X.loc[:, types.categorical]
    if target_col in features.columns:
        features = features.drop(target_col, axis=1)

    if features.shape[1] == 0:
        return

    features = features.astype('category')

    show_top = _get_n_top(features, "categorical")

    # can't use OrdinalEncoder because we might have mix of int and string
    ordinal_encoded = features.apply(lambda x: x.cat.codes)
    target = X[target_col]
    f = mutual_info_classif(
        ordinal_encoded, target,
        discrete_features=np.ones(X.shape[1], dtype=bool))
    top_k = np.argsort(f)[-show_top:][::-1]
    # large number of categories -> taller plot
    row_height = 3 if X.nunique().max() <= 5 else 5
    fig, axes = _make_subplots(n_plots=show_top, row_height=row_height)
    # FIXME mosaic doesn't like constraint layout?
    plt.suptitle("Categorical Features vs Target", y=1.02)
    for i, (col_ind, ax) in enumerate(zip(top_k, axes.ravel())):
        col = features.columns[col_ind]
        X_new = _prune_category_make_X(X, col, target_col)
        if kind == 'proportion':
            df = (X_new.groupby(col)[target_col]
                  .value_counts(normalize=True)
                  .unstack()
                  .sort_values(by=target[0]))  # hacky way to get a class name
            df.plot(kind='barh', stacked='True', ax=ax, legend=i == 0)
            ax.set_title(col)
            ax.set_ylabel(None)
        elif kind == 'mosaic':
            warn("Mosaic plots are buggy right now, come back later.",
                 UserWarning)
            # This seems pretty broken, abandoning for now
            # counts = pd.crosstab(X_new[col], X_new[target_col])

            mosaic(X_new, [col, target_col],
                   horizontal=False, ax=ax)
            # ,
            # labelizer=lambda k: counts.loc[k[0], k[1]])
        elif kind == 'count':
            # absolute counts
            # FIXME show f value
            # FIXME shorten titles?
            sns.countplot(y=col, data=X_new, ax=ax, hue=target_col)
        else:
            raise ValueError("Unknown plot kind {}".format(kind))
        _short_tick_names(ax)

    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()


def plot_supervised(X, target_col, types=None, verbose=10):
    """Exploration plots for classification and regression.

    Determines whether the target is categorical or continuous and plots the
    target distribution. Then calls the relevant plotting functions
    accordingly.


    Parameters
    ----------
    X : dataframe
        Input data including features and target
    target_col : str or int
        Identifier of the target column in X
    types : dataframe of types, optional.
        Output of detect_types on X. Can be used to avoid recomputing the
        types.
    verbose : int, default=10
        Controls the verbosity (output).

    See also
    --------
    plot_regression_continuous
    plot_regression_categorical
    plot_classification_continuous
    plot_classification_categorical
    """
    if types is None:
        types = detect_types(X)
    # aggressively low_cardinality integers plot better as categorical
    if types.low_card_int.any():
        for col in types.index[types.low_card_int]:
            # yes we con't need a loop
            types.loc[col, 'low_card_int'] = False
            types.loc[col, 'categorical'] = True

    # if any dirty floats, tell user to clean them first
    if types.dirty_float.any():
        warn("Found some dirty floats! "
             "Clean em up first:\n{}".format(
                 types.index[types.dirty_float]),
             UserWarning)

    if types.continuous[target_col]:
        print("Target looks like regression")
        # regression
        # make sure we include the target column in X
        # even though it's not categorical
        plt.figure()
        plt.hist(X[target_col], bins='auto')
        plt.xlabel(target_col)
        plt.ylabel("frequency")
        plt.title("Target distribution")
        plot_regression_continuous(X, target_col, types=types)
        plot_regression_categorical(X, target_col, types=types)
    else:
        print("Target looks like classification")
        # regression
        # make sure we include the target column in X
        # even though it's not categorical
        plt.figure()
        X[target_col].value_counts().plot(kind='barh', ax=plt.gca())
        plt.title("Target distribution")
        plt.ylabel("Label")
        plt.xlabel("Count")
        plot_classification_continuous(X, target_col, types=types)
        plot_classification_categorical(X, target_col, types=types)
