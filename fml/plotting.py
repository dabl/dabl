from .preprocessing import detect_types_dataframe

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from warnings import warn


def plot_continuous_unsupervised(X):
    pass


def plot_categorical_unsupervised(X):
    pass


def _shortname(some_string, maxlen=20):
    if len(some_string) > maxlen:
        return some_string[:maxlen - 3] + "..."
    else:
        return some_string


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
        X_new = X
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
    types = detect_types_dataframe(X)
    # if any dirty floats, tell user to clean them first
    plot_continuous_unsupervised(X.loc[:, types.continous])
    plot_categorical_unsupervised(X.loc[:, types.categorical])


def plot_regression_continuous(X, target_col):
    features = X.drop(target_col, axis=1)
    show_top = _get_n_top(features, "continuous")

    target = X[target_col]
    # HACK we should drop them per column before feeding them into f_regression
    # FIXME
    features_imp = SimpleImputer().fit_transform(features)
    f, p = f_regression(features_imp, target)
    top_k = np.argsort(f)[-show_top:][::-1]
    # we could do better lol
    n_cols = min(5, features.shape[1])
    n_rows = int(np.ceil(show_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows),
                             constrained_layout=True)
    # FIXME this could be a function or maybe using seaborn
    plt.suptitle("Continuous Feature vs Target")
    for i, (col, ax) in enumerate(zip(top_k, axes.ravel())):
        if i % n_cols == 0:
            ax.set_ylabel(target_col)
        ax.plot(features.iloc[:, col], target, 'o', alpha=.6)
        ax.set_xlabel(_shortname(features.columns[col]))
        ax.set_title("F={:.2E}".format(f[col]))

    for j in range(i + 1, n_rows * n_cols):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()


def plot_regression_categorical(X, target_col):
    X = X.copy()
    features = X.drop(target_col, axis=1)

    show_top = _get_n_top(features, "categorical")
    for col in X.columns:
        if col != target_col:
            X[col] = X[col].astype("category")
            # seaborn needs to know these are categories
    # can't use OrdinalEncoder because we might have mix of int and string
    ordinal_encoded = features.apply(lambda x: x.cat.codes)
    target = X[target_col]
    f = mutual_info_regression(
        ordinal_encoded, target,
        discrete_features=np.ones(X.shape[1], dtype=bool))
    top_k = np.argsort(f)[-show_top:][::-1]
    n_cols = min(5, features.shape[1])
    n_rows = int(np.ceil(show_top / n_cols))
    max_levels = X.nunique().max()
    if max_levels <= 5:
        height = 3
    else:
        height = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, height * n_rows),
                             constrained_layout=True)
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

    for j in range(i + 1, n_rows * n_cols):
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


def plot_classification_continuous(X, target_col):
    top_for_interactions = 20
    features = X.drop(target_col, axis=1)
    features_imp = SimpleImputer().fit_transform(features)
    target = X[target_col]
    # FIXME if one class only has NaN for a value we crash! :-/
    # TODO univariate plot?
    # already on diagonal for pairplot but not for many features
    if X.shape[1] <= 5:
        # for n_dim <= 5 we do full pairplot plot
        # FIXME filling in missing values here b/c of a bug in seaborn
        # we really shouldn't be doing this
        # https://github.com/mwaskom/seaborn/issues/1699
        X_imp = X.fillna(X.median(axis=0))
        sns.pairplot(X_imp, vars=[x for x in X.columns if x != target_col],
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

        g = sns.FacetGrid(df, col='variable', hue=target_col, col_wrap=5,
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
    if type(axes).__name__ == "AxesSubplot":
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


def plot_classification_categorical(X, target_col):
    X = X.copy()
    X = X.astype('category')
    features = X.drop(target_col, axis=1)
    show_top = _get_n_top(features, "categorical")

    # can't use OrdinalEncoder because we might have mix of int and string
    ordinal_encoded = features.apply(lambda x: x.cat.codes)
    target = X[target_col]
    f = mutual_info_classif(
        ordinal_encoded, target,
        discrete_features=np.ones(X.shape[1], dtype=bool))
    # FIXME copy & paste from regression, refactor
    top_k = np.argsort(f)[-show_top:][::-1]
    n_cols = min(5, features.shape[1])
    n_rows = int(np.ceil(show_top / n_cols))
    max_levels = X.nunique().max()
    if max_levels <= 5:
        height = 3
    else:
        height = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, height * n_rows),
                             constrained_layout=True)
    plt.suptitle("Categorical Feature vs Target")
    for i, (col_ind, ax) in enumerate(zip(top_k, axes.ravel())):
        col = features.columns[i]
        X_new = _prune_category_make_X(X, col, target_col)
        sns.countplot(y=col, data=X_new, ax=ax, hue=target_col)
        _short_tick_names(ax)

    for j in range(i + 1, n_rows * n_cols):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()


def _get_n_top(features, name):
    if features.shape[1] > 20:
        print("Showing only top 10 of {} {} features".format(
            features.shape[1], name))
        # too many features, show just top 10
        show_top = 10
    else:
        show_top = features.shape[1]
    return show_top


def plot_supervised(X, target_col, verbose=10):
    types = detect_types_dataframe(X)
    features = [c for c in X.columns if c != target_col]
    # if any dirty floats, tell user to clean them first
    if types.dirty_float.any():
        warn("Found some dirty floats! "
             "Clean em up first:\n{}".format(
                 types.index[types.dirty_float]),
             UserWarning)
    if types.low_card_int[features].any():
        warn("Found some low cardinality ints. "
             "No idea what to do, ignoring for now:\n{}".format(
                 types.index[types.low_card_int]),
             UserWarning)
    if types.continuous[target_col]:
        print("Target looks like regression")
        # regression
        # make sure we include the target column in X
        # even though it's not categorical
        plt.hist(X[target_col], bins='auto')
        plt.xlabel(target_col)
        plt.ylabel("frequency")
        plt.title("Target distribution")
        mask_for_categorical = types.categorical.copy()
        mask_for_categorical[target_col] = True
        plot_regression_continuous(X.loc[:, types.continuous], target_col)
        plot_regression_categorical(X.loc[:, mask_for_categorical], target_col)
    else:
        print("Target looks like classification")
        # regression
        # make sure we include the target column in X
        # even though it's not categorical
        X[target_col].value_counts().plot(kind='barh')
        plt.title("Target distribution")
        plt.ylabel("Label")
        plt.xlabel("Count")
        if types.continuous.any():
            mask_for_continuous = types.continuous.copy()
            mask_for_continuous[target_col] = True
            plot_classification_continuous(X.loc[:, mask_for_continuous],
                                           target_col)
        plot_classification_categorical(X.loc[:, types.categorical],
                                        target_col)
