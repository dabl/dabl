import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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
from sklearn.metrics import recall_score

from ..preprocessing import detect_types, clean, guess_ordinal
from .utils import (_check_X_target_col, _get_n_top, _make_subplots,
                    _short_tick_names, _shortname, _prune_category_make_X,
                    find_pretty_grid, _find_scatter_plots_classification,
                    class_hists, discrete_scatter, mosaic_plot,
                    _find_inliers, pairplot, _get_scatter_alpha,
                    _get_scatter_size)
from warnings import warn


def plot_regression_continuous(X, target_col, types=None,
                               scatter_alpha='auto', scatter_size='auto',
                               drop_outliers=True, **kwargs):
    """Plots for continuous features in regression.

    Creates plots of all the continuous features vs the target.
    Relevant features are determined using F statistics.

    Parameters
    ----------
    X : dataframe
        Input data including features and target.
    target_col : str or int
        Identifier of the target column in X.
    types : dataframe of types, optional
        Output of detect_types on X. Can be used to avoid recomputing the
        types.
    scatter_alpha : float, default='auto'
        Alpha values for scatter plots. 'auto' is dirty hacks.
    scatter_size : float, default='auto'
        Marker size for scatter plots. 'auto' is dirty hacks.
    drop_outliers : bool, default=True
        Whether to drop outliers when plotting.
    """
    types = _check_X_target_col(X, target_col, types, task="regression")

    if np.isnan(X[target_col]).any():
        X = X.dropna(subset=[target_col])
        warn("Missing values in target_col have been removed for regression",
             UserWarning)

    if drop_outliers:
        inliers = _find_inliers(X.loc[:, target_col])
        X = X.loc[inliers, :]

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
    scatter_alpha = _get_scatter_alpha(scatter_alpha, X[target_col])
    scatter_size = _get_scatter_size(scatter_size, X[target_col])
    for i, (col_idx, ax) in enumerate(zip(top_k, axes.ravel())):
        if i % axes.shape[1] == 0:
            ax.set_ylabel(_shortname(target_col))
        col = features.columns[col_idx]
        if drop_outliers:
            inliers = _find_inliers(features.loc[:, col])
            ax.scatter(features.loc[inliers, col], target[inliers],
                       alpha=scatter_alpha, s=scatter_size)
        else:
            ax.scatter(features.loc[:, col], target,
                       alpha=scatter_alpha, s=scatter_size)
        ax.set_xlabel(_shortname(col))
        ax.set_title("F={:.2E}".format(f[col_idx]))

    for j in range(i + 1, axes.size):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()
    return axes


def plot_regression_categorical(X, target_col, types=None, **kwargs):
    """Plots for categorical features in regression.

    Creates box plots of target distribution for important categorical
    features. Relevant features are identified using mutual information.

    For high cardinality categorical variables (variables with many categories)
    only the most frequent categories are shown.

    Parameters
    ----------
    X : dataframe
        Input data including features and target.
    target_col : str or int
        Identifier of the target column in X.
    types : dataframe of types, optional
        Output of detect_types on X. Can be used to avoid recomputing the
        types.
    """
    types = _check_X_target_col(X, target_col, types, task="regression")

    # drop nans from target column
    if np.isnan(X[target_col]).any():
        X = X.dropna(subset=[target_col])
        warn("Missing values in target_col have been removed for regression",
             UserWarning)

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
                                   scatter_alpha='auto', scatter_size="auto",
                                   univariate_plot='histogram',
                                   drop_outliers=True, plot_pairwise=True,
                                   top_k_interactions=10, random_state=None,
                                   **kwargs):
    """Plots for continuous features in classification.

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
        Input data including features and target.
    target_col : str or int
        Identifier of the target column in X.
    types : dataframe of types, optional.
        Output of detect_types on X. Can be used to avoid recomputing the
        types.
    scatter_alpha : float, default='auto'
        Alpha values for scatter plots. 'auto' is dirty hacks.
    scatter_size : float, default='auto'
        Marker size for scatter plots. 'auto' is dirty hacks.
    univariate_plot : string, default="histogram"
        Supported: 'histogram' and 'kde'.
    drop_outliers : bool, default=True
        Whether to drop outliers when plotting.
    plot_pairwise : bool, default=True
        Whether to create pairwise plots. Can be a bit slow.
    top_k_interactions : int, default=10
        How many pairwise interactions to consider
        (ranked by univariate f scores).
        Runtime is quadratic in this, but higher numbers might find more
        interesting interactions.
    random_state : int, None or numpy RandomState
        Random state used for subsampling for determining pairwise features
        to show.

    Notes
    -----
    important kwargs parameters are: scatter_size and scatter_alpha.
    """

    types = _check_X_target_col(X, target_col, types, task='classification')

    features = X.loc[:, types.continuous]
    if target_col in features.columns:
        features = features.drop(target_col, axis=1)
    if features.shape[1] == 0:
        return

    features_imp = SimpleImputer().fit_transform(features)
    target = X[target_col]
    figures = []
    if features.shape[1] <= 5:
        pairplot(X, target_col=target_col, columns=features.columns,
                 scatter_alpha=scatter_alpha,
                 scatter_size=scatter_size)
        title = "Continuous features"
        if features.shape[1] > 1:
            title = title + " pairplot"
        plt.suptitle(title, y=1.02)

        fig = plt.gcf()
    else:
        # univariate plots
        f = _plot_univariate_classification(features, features_imp, target,
                                            drop_outliers, target_col,
                                            univariate_plot, hue_order)
        figures.append(plt.gcf())

        # FIXME remove "variable = " from title, add f score
        # pairwise plots
        if not plot_pairwise:
            return figures
        top_k = np.argsort(f)[-top_k_interactions:][::-1]
        fig, axes = _plot_top_pairs(features_imp[:, top_k], target,
                                    scatter_alpha, scatter_size,
                                    feature_names=features.columns[top_k],
                                    how_many=4, random_state=random_state)
        fig.suptitle("Top feature interactions")
    figures.append(fig)
    if not plot_pairwise:
        return figures
    # get some PCA directions
    # we're using all features here, not only most informative
    # should we use only those?
    n_components = min(top_k_interactions, features.shape[0],
                       features.shape[1])
    if n_components < 2:
        return figures
    features_scaled = _plot_pca_classification(
        n_components, features_imp, target, scatter_alpha, scatter_size,
        random_state=random_state)
    figures.append(plt.gcf())
    # LDA
    _plot_lda_classification(features_scaled, target, top_k_interactions,
                             scatter_alpha, scatter_size,
                             random_state=random_state)
    figures.append(plt.gcf())
    return figures


def _plot_pca_classification(n_components, features_imp, target,
                             scatter_alpha='auto', scatter_size='auto',
                             random_state=None):
    pca = PCA(n_components=n_components)
    features_scaled = scale(features_imp)
    features_pca = pca.fit_transform(features_scaled)
    feature_names = ['PCA {}'.format(i) for i in range(n_components)]
    fig, axes = _plot_top_pairs(features_pca, target, scatter_alpha,
                                scatter_size,
                                feature_names=feature_names,
                                how_many=3, additional_axes=1,
                                random_state=random_state)
    ax = axes.ravel()[-1]
    ax.plot(pca.explained_variance_ratio_, label='variance')
    ax.plot(np.cumsum(pca.explained_variance_ratio_),
            label='cumulative variance')
    ax.set_title("Scree plot (PCA explained variance)")
    ax.legend()
    fig.suptitle("Discriminating PCA directions")
    return features_scaled


def _plot_lda_classification(features, target, top_k_interactions,
                             scatter_alpha='auto', scatter_size='auto',
                             random_state=None):
    # assume features are scaled
    n_components = min(top_k_interactions, features.shape[0],
                       features.shape[1], target.nunique() - 1)
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    features_lda = lda.fit_transform(features, target)
    # we should probably do macro-average recall here as everywhere else?
    print("Linear Discriminant Analysis training set score: {:.3f}".format(
          recall_score(target, lda.predict(features), average='macro')))
    if features_lda.shape[1] < 2:
        # Do a single plot and exit
        plt.figure()
        single_lda = pd.DataFrame({'feature': features_lda.ravel(),
                                   'target': target})
        class_hists(single_lda, 'feature', 'target', legend=True)
        plt.title("Linear Discriminant")
        return
    feature_names = ['LDA {}'.format(i) for i in range(n_components)]

    fig, _ = _plot_top_pairs(features_lda, target, scatter_alpha, scatter_size,
                             feature_names=feature_names,
                             random_state=random_state)
    fig.suptitle("Discriminating LDA directions")


def _plot_top_pairs(features, target, scatter_alpha='auto',
                    scatter_size='auto',
                    feature_names=None, how_many=4, additional_axes=0,
                    random_state=None):
    top_pairs = _find_scatter_plots_classification(
        features, target, how_many=how_many, random_state=random_state)
    if feature_names is None:
        feature_names = ["feature {}".format(i)
                         for i in range(features.shape[1])]
    fig, axes = _make_subplots(len(top_pairs) + additional_axes, row_height=4)
    for x, y, score, ax in zip(top_pairs.feature0, top_pairs.feature1,
                               top_pairs.score, axes.ravel()):
        discrete_scatter(features[:, x], features[:, y],
                         c=target, ax=ax, alpha=scatter_alpha,
                         s=scatter_size)
        ax.set_xlabel(_shortname(feature_names[x]))
        ax.set_ylabel(_shortname(feature_names[y]))
        ax.set_title("{:.3f}".format(score))
    return fig, axes


def _plot_univariate_classification(features, features_imp, target,
                                    drop_outliers,
                                    target_col, univariate_plot, hue_order):
    # univariate plots
    show_top = _get_n_top(features, "continuous")
    f, p = f_classif(features_imp, target)
    top_k = np.argsort(f)[-show_top:][::-1]
    # FIXME this will fail if a feature is always
    # NaN for a particular class
    best_features = features.iloc[:, top_k].copy()

    if drop_outliers:
        for col in best_features.columns:
            inliers = _find_inliers(best_features.loc[:, col])
            best_features[~inliers] = np.NaN

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
        # row_height = 3 if target.nunique() < 5 else 5
        n_classes = target.nunique()
        row_height = n_classes * 1 if n_classes < 10 else n_classes * .5
        fig, axes = _make_subplots(n_plots=show_top, row_height=row_height)
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
    return f


def plot_classification_categorical(X, target_col, types=None, kind='auto',
                                    hue_order=None, **kwargs):
    """Plots for categorical features in classification.

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
        if X[target_col].nunique() > 5:
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
            mosaic_plot(X_new, col, target_col, ax=ax, legend=i == 0)
            ax.set_title(col)
        elif kind == 'count':
            X_new = _prune_category_make_X(X, col, target_col)

            # absolute counts
            # FIXME show f value
            # FIXME shorten titles?
            props = {}
            if X[target_col].nunique() > 15:
                props['font.size'] = 6
            with mpl.rc_context(props):
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


def plot(X, y=None, target_col=None, type_hints=None, scatter_alpha='auto',
         scatter_size='auto', verbose=10, plot_pairwise=True,
         **kwargs):
    """Automatic plots for classification and regression.

    Determines whether the target is categorical or continuous and plots the
    target distribution. Then calls the relevant plotting functions
    accordingly.


    Parameters
    ----------
    X : DataFrame
        Input features. If target_col is specified, X also includes the
        target.
    y : Series or numpy array, optional.
        Target. You need to specify either y or target_col.
    target_col : string or int, optional
        Column name of target if included in X.
    type_hints : dict or None
        If dict, provide type information for columns.
        Keys are column names, values are types as provided by detect_types.
    scatter_alpha : float, default='auto'
        Alpha values for scatter plots. 'auto' is dirty hacks.
    scatter_size : float, default='auto'.
        Marker size for scatter plots. 'auto' is dirty hacks.
    plot_pairwise : bool, default=True
        Whether to include pairwise scatterplots for classification.
        These can be somewhat expensive to compute.
    verbose : int, default=10
        Controls the verbosity (output).

    See also
    --------
    plot_regression_continuous
    plot_regression_categorical
    plot_classification_continuous
    plot_classification_categorical
    """
    if ((y is None and target_col is None)
            or (y is not None) and (target_col is not None)):
        raise ValueError(
            "Need to specify either y or target_col.")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, str):
        warnings.warn("The second positional argument of plot is a Series 'y'."
                      " If passing a column name, use a keyword.",
                      FutureWarning)
        target_col = y
        y = None
    if target_col is None:
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if y.name is None:
            y = y.rename('target')
        target_col = y.name
        X = pd.concat([X, y], axis=1)

    X, types = clean(X, type_hints=type_hints, return_types=True,
                     target_col=target_col)
    types = _check_X_target_col(X, target_col, types=types)
    # low_cardinality integers plot better as categorical
    # FIXME the logic should be down in the plotting functions maybe
    # or at least passed on so we can do better.
    if types.low_card_int.any():
        for col in types.index[types.low_card_int]:
            # kinda hacky for now
            if guess_ordinal(X[col]):
                types.loc[col, 'low_card_int'] = False
                types.loc[col, 'continuous'] = True
            else:
                types.loc[col, 'low_card_int'] = False
                types.loc[col, 'categorical'] = True

    if types.continuous[target_col]:
        print("Target looks like regression")
        # FIXME we might be overwriting the original dataframe here?
        X[target_col] = X[target_col].astype(np.float)
        # regression
        # make sure we include the target column in X
        # even though it's not categorical
        plt.hist(X[target_col], bins='auto')
        plt.xlabel(_shortname(target_col))
        plt.ylabel("frequency")
        plt.title("Target distribution")
        plot_regression_continuous(X, target_col, types=types,
                                   scatter_alpha=scatter_alpha,
                                   scatter_size=scatter_size, **kwargs)
        plot_regression_categorical(X, target_col, types=types, **kwargs)
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
        if len(counts) >= 50:
            print("Not plotting anything for 50 classes or more."
                  "Current visualizations are quite useless for"
                  " this many classes. Try slicing the data.")
        plot_classification_continuous(
            X, target_col, types=types, hue_order=counts.index,
            scatter_alpha=scatter_alpha, scatter_size=scatter_size,
            plot_pairwise=plot_pairwise, **kwargs)
        plot_classification_categorical(X, target_col, types=types,
                                        hue_order=counts.index, **kwargs)
