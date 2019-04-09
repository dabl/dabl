import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from sklearn.feature_selection import (f_regression,
                                       mutual_info_regression,
                                       mutual_info_classif,
                                       f_classif)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ..preprocessing import detect_types, clean
from .utils import (_check_X_target_col, _get_n_top, _make_subplots,
                    _short_tick_names, _shortname, _prune_category_make_X,
                    find_pretty_grid, _find_scatter_plots_classification,
                    class_hists, discrete_scatter, mosaic_plot)


def plot_regression_continuous(X, target_col, types=None,
                               scatter_alpha=1.):
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
    scatter_alpha : float, default=1.
        Alpha values for scatter plots.
    """
    types = _check_X_target_col(X, target_col, types, task="regression")

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
        ax.plot(features.iloc[:, col], target, 'o',
                alpha=scatter_alpha)
        ax.set_xlabel(_shortname(features.columns[col]))
        ax.set_title("F={:.2E}".format(f[col]))

    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()


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
    types = _check_X_target_col(X, target_col, types, task="regression")

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


def plot_classification_continuous(X, target_col, types=None, hue_order=None,
                                   scatter_alpha=1.,
                                   univariate_plot='histogram'):
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
    scatter_alpha : float, default=1.
        Alpha values for scatter plots.
    univariate_plot : string, default="histogram"
        Supported: 'histogram' and 'kde'.
    """
    types = _check_X_target_col(X, target_col, types, task='classification')

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
        # FIXME we should use "univariate_plot" here
        sns.pairplot(X_imp, vars=features.columns,
                     hue=target_col, diag_kind='hist',
                     plot_kws={'alpha': scatter_alpha})
        plt.suptitle("Continuous features pairplot", y=1.02)
    else:
        # univariate plots
        show_top = _get_n_top(features, "continuous")
        f, p = f_classif(features_imp, target)
        top_k = np.argsort(f)[-show_top:][::-1]
        # FIXME this will fail if a feature is always
        # NaN for a particular class
        best_features = features.iloc[:, top_k].copy()

        best_features[target_col] = target
        if univariate_plot == 'kde':
            df = best_features.melt(target_col)
            rows, cols = find_pretty_grid(show_top)

            g = sns.FacetGrid(df, col='variable', hue=target_col,
                              col_wrap=cols,
                              sharey=False, sharex=False, hue_order=hue_order)
            g = g.map(sns.kdeplot, "value", shade=True)
            g.axes[0].legend()
            plt.suptitle("Continuous features by target", y=1.02)
        elif univariate_plot == 'histogram':
            fig, axes = _make_subplots(n_plots=show_top)
            for i, (ind, ax) in enumerate(zip(top_k, axes.ravel())):
                class_hists(best_features, best_features.columns[i],
                            target_col, ax=ax, legend=i == 0)
                ax.set_title("F={:.2E}".format(f[ind]))
            for j in range(i + 1, axes.size):
                # turn off axis if we didn't fill last row
                axes.ravel()[j].set_axis_off()
        else:
            raise ValueError("Unknown value for univariate_plot: ",
                             univariate_plot)

        # FIXME remove "variable = " from title, add f score

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
            discrete_scatter(features_imp[:, i], features_imp[:, j],
                             c=target, ax=ax, alpha=scatter_alpha)
            ax.set_xlabel(features.columns[i])
            ax.set_ylabel(features.columns[j])
            ax.set_title("{:.3f}".format(score))
        fig.suptitle("Top feature interactions")
    # get some PCA directions
    # we're using all features here, not only most informative
    # should we use only those?
    n_components = min(top_for_interactions, features.shape[0],
                       features.shape[1])
    if n_components < 2:
        return
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

        discrete_scatter(features_pca[:, x], features_pca[:, y],
                         c=target, ax=ax, alpha=scatter_alpha)
        ax.set_xlabel("PCA {}".format(x))
        ax.set_ylabel("PCA {}".format(y))
        ax.set_title("{:.3f}".format(score))
    fig.suptitle("Discriminating PCA directions")
    # LDA
    if target.nunique() < 3:
        # lda currently only 2d scatter plot.
        return
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

        discrete_scatter(features_lda[:, x], features_lda[:, y],
                         c=target, ax=ax, alpha=scatter_alpha)
        ax.set_xlabel("LDA {}".format(x))
        ax.set_ylabel("LDA {}".format(y))
        ax.set_title("{:.3f}".format(score))
    fig.suptitle("Discriminating LDA directions")
    # TODO fancy manifolds?


def plot_classification_categorical(X, target_col, types=None, kind='auto',
                                    hue_order=None):
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
    kind : string, default 'auto'
        Kind of plot to show. Options are 'count', 'proportion',
        'mosaic' and 'auto'.
        Count shows raw class counts within categories
        (can be hard to read with imbalanced classes)
        Proportion shows class proportions within categories
        (can be misleading with imbalanced categories)
        Mosaic shows both aspects, but can be a bit busy.
        Auto uses mosaic plots for binary classification and counts otherwise.

    """
    types = _check_X_target_col(X, target_col, types, task="classification")
    if kind == "auto":
        if X[target_col].nunique() > 2:
            kind = 'count'
        else:
            kind = 'mosaic'

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
    row_height = 3 if features.nunique().max() <= 5 else 5
    fig, axes = _make_subplots(n_plots=show_top, row_height=row_height)
    plt.suptitle("Categorical Features vs Target", y=1.02)
    for i, (col_ind, ax) in enumerate(zip(top_k, axes.ravel())):
        col = features.columns[col_ind]
        if kind == 'proportion':
            X_new = _prune_category_make_X(X, col, target_col)

            df = (X_new.groupby(col)[target_col]
                  .value_counts(normalize=True)
                  .unstack()
                  .sort_values(by=target[0]))  # hacky way to get a class name
            df.plot(kind='barh', stacked='True', ax=ax, legend=i == 0)
            ax.set_title(col)
            ax.set_ylabel(None)
        elif kind == 'mosaic':
            # how many categories make up at least 1% of data:
            n_cats = (X[col].value_counts() / len(X) > 0.01).sum()
            n_cats = np.minimum(n_cats, 20)
            X_new = _prune_category_make_X(X, col, target_col,
                                           max_categories=n_cats)
            mosaic_plot(X_new, col, target_col, ax=ax)
            ax.set_title(col)
        elif kind == 'count':
            X_new = _prune_category_make_X(X, col, target_col)

            # absolute counts
            # FIXME show f value
            # FIXME shorten titles?
            sns.countplot(y=col, data=X_new, ax=ax, hue=target_col,
                          hue_order=hue_order)
            if i > 0:
                ax.legend(())
        else:
            raise ValueError("Unknown plot kind {}".format(kind))
        _short_tick_names(ax)

    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()


def plot_supervised(X, target_col, type_hints=None, scatter_alpha=1.,
                    verbose=10):
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
    type_hints : dict or None
        If dict, provide type information for columns.
        Keys are column names, values are types as provided by detect_types.
    scatter_alpha : float, default=1.
        Alpha values for scatter plots.
    verbose : int, default=10
        Controls the verbosity (output).

    See also
    --------
    plot_regression_continuous
    plot_regression_categorical
    plot_classification_continuous
    plot_classification_categorical
    """
    X = clean(X, type_hints=type_hints)
    # recompute types after cleaning:
    types = _check_X_target_col(X, target_col, type_hints=type_hints)
    # low_cardinality integers plot better as categorical
    if types.low_card_int.any():
        for col in types.index[types.low_card_int]:
            if col == target_col:
                continue
            # kinda hacky for now
            if X[col].nunique() < 20:
                types.loc[col, 'low_card_int'] = False
                types.loc[col, 'categorical'] = True
            else:
                types.loc[col, 'low_card_int'] = False
                types.loc[col, 'continuous'] = True

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
        plot_regression_continuous(X, target_col, types=types,
                                   scatter_alpha=scatter_alpha)
        plot_regression_categorical(X, target_col, types=types)
    else:
        print("Target looks like classification")
        # regression
        # make sure we include the target column in X
        # even though it's not categorical
        plt.figure()
        counts = pd.DataFrame(X[target_col].value_counts())
        melted = counts.T.melt().rename(
            columns={'variable': 'class', 'value': 'count'})
        # class could be a string that's a float
        # seaborn is trying to be smart unless we declare it categorical
        # we actually fixed counts to have categorical index
        # but melt destroys it:
        # https://github.com/pandas-dev/pandas/issues/15853
        melted['class'] = melted['class'].astype('category')
        sns.barplot(y='class', x='count', data=melted)
        plt.title("Target distribution")
        plot_classification_continuous(X, target_col, types=types,
                                       hue_order=counts.index,
                                       scatter_alpha=scatter_alpha)
        plot_classification_categorical(X, target_col, types=types,
                                        hue_order=counts.index)
