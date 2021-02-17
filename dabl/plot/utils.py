from warnings import warn
from functools import reduce
import itertools


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from seaborn.utils import despine


# from sklearn.dummy import DummyClassifier
# from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit


from ..preprocessing import detect_types
from .._config import get_config


def find_pretty_grid(n_plots, max_cols=5):
    """Determine a good grid shape for subplots.

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


def plot_coefficients(coefficients, feature_names, n_top_features=10,
                      classname=None, ax=None):
    """Visualize coefficients of a linear model.

    Parameters
    ----------
    coefficients : nd-array, shape (n_features,)
        Model coefficients.

    feature_names : list or nd-array of strings, shape (n_features,)
        Feature names for labeling the coefficients.

    n_top_features : int, default=10
        How many features to show. The function will show the largest (most
        positive) and smallest (most negative)  n_top_features coefficients,
        for a total of 2 * n_top_features coefficients.
    """

    coefficients = coefficients.squeeze()
    feature_names = np.asarray(feature_names)
    if coefficients.ndim > 1:
        # this is not a row or column vector
        raise ValueError("coefficients must be 1d array or column vector, got"
                         " shape {}".format(coefficients.shape))
    coefficients = coefficients.ravel()

    if len(coefficients) != len(feature_names):
        raise ValueError("Number of coefficients {} doesn't match number of"
                         "feature names {}.".format(len(coefficients),
                                                    len(feature_names)))
    # get coefficients with large absolute values
    coef = coefficients.ravel()
    mask = coef != 0
    coef = coef[mask]
    feature_names = feature_names[mask]
    # FIXME this could be easier with pandas by sorting by a column
    interesting_coefficients = np.argsort(np.abs(coef))[-n_top_features:]
    new_inds = np.argsort(coef[interesting_coefficients])
    interesting_coefficients = interesting_coefficients[new_inds]
    # plot them
    if ax is None:
        plt.figure(figsize=(len(interesting_coefficients), 5))
        ax = plt.gca()
    colors = ['red' if c < 0 else 'blue'
              for c in coef[interesting_coefficients]]
    ax.bar(np.arange(len(interesting_coefficients)),
           coef[interesting_coefficients],
           color=colors)
    feature_names = np.array(feature_names)
    ax.set_xticks(np.arange(0, len(interesting_coefficients)))
    ax.set_xticklabels(feature_names[interesting_coefficients],
                       rotation=60, ha="right")
    _short_tick_names(ax, ticklabel_length=20)
    ax.set_ylabel("Coefficient magnitude")
    ax.set_xlabel("Feature")
    ax.set_title(classname)
    return feature_names[interesting_coefficients]


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f", origin='lower'):
    if ax is None:
        ax = plt.gca()
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(_shortname(xlabel, maxlen=40))
    ax.set_ylabel(_shortname(ylabel, maxlen=40))
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    xticklabels = [_shortname(label, maxlen=40) for label in xticklabels]
    yticklabels = [_shortname(label, maxlen=40) for label in yticklabels]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)
    if origin == 'upper':
        ylim = ax.get_ylim()
        ax.set_ylim(ylim[::-1])

    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img


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
    if not get_config()['truncate_labels']:
        return some_string
    if len(some_string) > maxlen:
        return some_string[:maxlen - 3] + "..."
    else:
        return some_string


def mosaic_plot(data, rows, cols, vary_lightness=False, ax=None, legend=True):
    """Create a mosaic plot from a dataframe.

    Right now only horizontal mosaic plots are supported,
    i.e. rows are prioritized over columns.

    Parameters
    ----------
    data : pandas data frame
        Data to tabulate.
    rows : column specifier
        Column in data to tabulate across rows.
    cols : column specifier
        Column in data to use to subpartition rows.
    vary_lightness : bool, default=False
        Whether to vary lightness across categories.
    ax : matplotlib axes or None
        Axes to plot into.
    legend : boolean, default=True
        Whether to create a legend.

    Examples
    --------
    >>> from dabl.datasets import load_titanic
    >>> data = load_titanic()
    >>> mosaic_plot(data, 'sex', 'survived')
    """

    cont = pd.crosstab(data[cols], data[rows])
    sort = np.argsort((cont / cont.sum()).iloc[0])
    cont = cont.iloc[:, sort]
    if ax is None:
        ax = plt.gca()
    pos_y = 0
    positions_y = []
    n_cols = cont.shape[1]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, col in enumerate(cont.columns):
        height = cont[col].sum()
        positions_y.append(pos_y + height / 2)

        pos_x = 0
        for j, row in enumerate(cont[col]):
            width = row / height
            color = colors[j]
            if vary_lightness:
                color = _lighten_color(color, (i + 1) / (n_cols + 1))
            rect = Rectangle((pos_x, pos_y), width, height, edgecolor='k',
                             facecolor=color)
            pos_x += width
            ax.add_patch(rect)
        pos_y += height

    if legend:
        legend_elements = [Patch(facecolor=colors[i], edgecolor='k')
                           for i in range(len(cont.index))]
        legend_labels = [str(index) for index in cont.index]
        ax.legend(legend_elements, legend_labels)

    ax.set_ylim(0, pos_y)
    ax.set_yticks(positions_y)
    ax.set_yticklabels(cont.columns)


def _lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    c = color
    amount += 0.5
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


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
    res = res.cat.add_categories(['dabl_other']).fillna("dabl_other")
    return res


def _prune_category_make_X(X, col, target_col, max_categories=20):
    col_values = X[col]
    if col_values.nunique() > max_categories:
        # keep only top 10 categories if there are more than 20
        col_values = _prune_categories(col_values,
                                       max_categories=min(10, max_categories))
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
            X[col].fillna("dabl_missing", inplace=True)
        else:
            X[col].fillna(max_value + 1, inplace=True)
    return X


def _make_subplots(n_plots, max_cols=5, row_height=3):
    """Create a harmonious subplot grid.
    """
    n_rows, n_cols = find_pretty_grid(n_plots, max_cols=max_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, row_height * n_rows),
                             constrained_layout=True)
    # we don't want ravel to fail, this is awkward!
    axes = np.atleast_2d(axes)
    return fig, axes


def _check_X_target_col(X, target_col, types=None, type_hints=None, task=None):
    if types is None:
        types = detect_types(X, type_hints=type_hints)
    if (not isinstance(target_col, str) and hasattr(target_col, '__len__') and
            len(target_col) > 1):
        raise ValueError("target_col should be a column in X, "
                         "got {}".format(target_col))
    if target_col not in X.columns:
        raise ValueError("{} is not a valid column of X".format(target_col))

    if X[target_col].nunique() < 2:
        raise ValueError("Less than two classes present, {}, need at least two"
                         " for classification.".format(X.loc[0, target_col]))
    # FIXME we get target types here with detect_types,
    # but in the estimator with type_of_target
    if task == "classification" and not types.loc[target_col, 'categorical']:
        raise ValueError("Type for target column {} detected as {},"
                         " need categorical for classification.".format(
                             target_col, types.T.idxmax()[target_col]))
    if task == "regression" and (not types.loc[target_col, 'continuous']):
        raise ValueError("Type for target column {} detected as {},"
                         " need continuous for regression.".format(
                             target_col, types.T.idxmax()[target_col]))
    return types


def _short_tick_names(ax, label_length=20, ticklabel_length=10):
    """Shorten axes labels and tick labels.

    Uses _shortname to change labels as a side effect.

    Parameters
    ----------
    ax : matplotlib axes
        Axes on which to shorten labels.
    label_length : int, default=20
        Length of xlabel and ylabel
    ticklabel_length : int, default=10
        Length of each label in xticklabels and yticklabels
    """
    ax.set_xticklabels(
        [_shortname(t.get_text(), maxlen=ticklabel_length)
         for t in ax.get_xticklabels()]
    )
    ax.set_yticklabels(
        [_shortname(t.get_text(), maxlen=ticklabel_length)
         for t in ax.get_yticklabels()]
    )
    ax.set_xlabel(_shortname(ax.get_xlabel(), maxlen=label_length))
    ax.set_ylabel(_shortname(ax.get_ylabel(), maxlen=label_length))


def _find_scatter_plots_classification(X, target, how_many=3,
                                       random_state=None):
    # input is continuous
    # look at all pairs of features, find most promising ones
    # dummy = DummyClassifier(strategy='prior').fit(X, target)
    # baseline_score = recall_score(target, dummy.predict(X), average='macro')
    scores = []
    # converting to int here might save some time
    _, target = np.unique(target, return_inverse=True)
    # limit to 2000 training points for speed?
    train_size = min(2000, int(.9 * X.shape[0]))
    cv = StratifiedShuffleSplit(n_splits=3, train_size=train_size,
                                random_state=random_state)
    for i, j in itertools.combinations(np.arange(X.shape[1]), 2):
        this_X = X[:, [i, j]]
        # assume this tree is simple enough so not be able to overfit in 2d
        # so we don't bother with train/test split
        tree = DecisionTreeClassifier(max_leaf_nodes=8)
        scores.append((i, j, np.mean(cross_val_score(
            tree, this_X, target, cv=cv, scoring='recall_macro'))))

    scores = pd.DataFrame(scores, columns=['feature0', 'feature1', 'score'])
    top = scores.sort_values(by='score').iloc[-how_many:][::-1]
    return top


def discrete_scatter(x, y, c, unique_c=None, legend='first',
                     clip_outliers=True,
                     alpha='auto', s='auto', ax=None, **kwargs):
    """Scatter plot for categories.

    Creates a scatter plot for x and y grouped by c.


    Parameters
    ----------
    x : array-like
        x coordinates to scatter.
    y : array-like
        y coordinates to scatter.
    c : array-like
        Grouping of samples (similar to hue in seaborn).
    unique_c : array-like, default='None'
        Unique values of c considered in scatter. If not
        provided unique elements of c are determined.
    legend : bool, or "first", default="first"
        Whether to create a legend. "first" mean only the
        first one in a given gridspec.
    clip_outliers : bool, default='True'
        Whether to clip outliers in x and y. The limits are
        determined based on 0.01 and 0.99 quantiles of x and
        y ignoring nan values.
    alpha : float, default='auto'
        Alpha values for scatter plots. 'auto' is dirty hacks.
    s : float, default='auto'.
        Marker size for scatter plots. 'auto' is dirty hacks.
    ax : matplotlib axes, default=None
        Axes to plot into.
    kwargs :
        Passed through to plt.scatter.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from dabl.datasets import load_ames
    >>> data = load_ames()
    >>> fig = plt.figure()
    >>> discrete_scatter(
    ...    x=data["Year Built"],
    ...    y=data["SalePrice"],
    ...    c=data["Overall Qual"],
    ...    unique_c=[2, 4, 6, 8, 10],
    ...    legend=True,
    ...    alpha=0.3
    ... )
    """
    alpha = _get_scatter_alpha(alpha, x)
    s = _get_scatter_size(s, x)
    if ax is None:
        ax = plt.gca()
    if legend == "first":
        legend = (ax.get_geometry()[2] == 1)
    if unique_c is None:
        unique_c = np.unique(c)
    for i in unique_c:
        mask = c == i
        ax.scatter(x[mask], y[mask], label=i, s=s, alpha=alpha, **kwargs)
    if clip_outliers:
        x_low, x_high = _inlier_range(x)
        y_low, y_high = _inlier_range(y)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.set_xlim(max(x_low, xlims[0]), min(x_high, xlims[1]))
        ax.set_ylim(max(y_low, ylims[0]), min(y_high, ylims[1]))

    if legend:
        props = {}
        if len(unique_c) > 15:
            props['size'] = 6
        legend = ax.legend(prop=props)
        for handle in legend.legendHandles:
            handle.set_alpha(1)
            handle.set_sizes((100,))


def class_hists(data, column, target, bins="auto", ax=None, legend=True,
                scale_separately=True):
    """Grouped univariate histograms.

    Parameters
    ----------
    data : pandas DataFrame
        Input data to plot.
    column : column specifier
        Column in the data to compute histograms over (must be continuous).
    target : column specifier
        Target column in data, must be categorical.
    bins : string, int or array-like
        Number of bins, 'auto' or bin edges. Passed to np.histogram_bin_edges.
        We always show at least 5 bins for now.
    ax : matplotlib axes
        Axes to plot into.
    legend : boolean, default=True
        Whether to create a legend.
    scale_separately : boolean, default=True
        Whether to scale each class separately.

    Examples
    --------
    >>> from dabl.datasets import load_adult
    >>> data = load_adult()
    >>> class_hists(data, "age", "gender", legend=True)
    <AxesSubplot:xlabel='age'>
    """
    col_data = data[column].dropna()

    if ax is None:
        ax = plt.gca()
    if col_data.nunique() > 10:
        ordinal = False
        # histograms
        bin_edges = np.histogram_bin_edges(col_data, bins=bins)
        if len(bin_edges) > 30:
            bin_edges = np.histogram_bin_edges(col_data, bins=30)

        counts = {}
        for name, group in data.groupby(target)[column]:
            this_counts, _ = np.histogram(group, bins=bin_edges)
            counts[name] = this_counts
        counts = pd.DataFrame(counts)
    else:
        ordinal = True
        # ordinal data, count distinct values
        counts = data.groupby(target)[column].value_counts().unstack(target)
    if scale_separately:
        # normalize by maximum
        counts = counts / counts.max()
    bottom = counts.max().max() * 1.1
    for i, name in enumerate(counts.columns):
        if ordinal:
            ax.bar(range(counts.shape[0]), counts[name], width=.9,
                   bottom=bottom * i, tick_label=counts.index, linewidth=2,
                   edgecolor='k', label=name)
            xmin, xmax = 0 - .5, counts.shape[0] - .5
        else:
            ax.bar(bin_edges[:-1], counts[name], bottom=bottom * i, label=name,
                   align='edge', width=(bin_edges[1] - bin_edges[0]) * .9)
            xmin, xmax = bin_edges[0], bin_edges[-1]
        ax.hlines(bottom * i, xmin=xmin, xmax=xmax,
                  linewidth=1)
    if legend:
        ax.legend()
    ax.set_yticks(())
    ax.set_xlabel(_shortname(column))
    return ax


def pairplot(data, target_col, columns=None, scatter_alpha='auto',
             scatter_size='auto'):
    """Pairplot (scattermatrix)

    Because there's already too many implementations of this.
    This is meant for classification only.
    This is very bare-bones right now :-/

    Parameters
    ----------
    data : pandas dataframe
        Input data
    target_col : column specifier
        Target column in data.
    columns : column specifiers, default=None.
        Columns in data to include. None means all.
    scatter_alpha : float, default='auto'
        Alpha values for scatter plots. 'auto' is dirty hacks.
    scatter_size : float, default='auto'.
        Marker size for scatter plots. 'auto' is dirty hacks.
    """
    if columns is None:
        columns = data.columns.drop(target_col)
    n_features = len(columns)
    fig, axes = plt.subplots(n_features, n_features,
                             figsize=(n_features * 3, n_features * 3))
    axes = np.atleast_2d(axes)
    for ax, (i, j) in zip(axes.ravel(),
                          itertools.product(range(n_features), repeat=2)):
        legend = i == 0 and j == n_features - 1
        if i == j:
            class_hists(data, columns[i], target_col, ax=ax.twinx())
        else:
            discrete_scatter(data[columns[j]], data[columns[i]],
                             c=data[target_col], legend=legend, ax=ax,
                             alpha=scatter_alpha,
                             s=scatter_size)
        if j == 0:
            ax.set_ylabel(columns[i])
        else:
            ax.set_ylabel("")
            ax.set_yticklabels(())
        if i == n_features - 1:
            ax.set_xlabel(_shortname(columns[j]))
        else:
            ax.set_xlabel("")
            ax.set_xticklabels(())
    despine(fig)
    if n_features > 1:
        axes[0, 0].set_yticks(axes[0, 1].get_yticks())
        axes[0, 0].set_ylim(axes[0, 1].get_ylim())
    return axes


def _inlier_range(series):
    low = np.nanquantile(series, 0.01)
    high = np.nanquantile(series, 0.99)
    assert low <= high
    # the two is a complete hack
    inner_range = (high - low) / 2
    return low - inner_range, high + inner_range


def _find_inliers(series):
    low, high = _inlier_range(series)
    mask = series.between(low, high)
    mask = mask | series.isna()
    dropped = len(mask) - mask.sum()
    if dropped > 0:
        warn("Dropped {} outliers in column {}.".format(
            int(dropped), series.name), UserWarning)
    return mask


def _clean_outliers(data):
    def _find_outliers_series(series):
        series = series.dropna()
        low = series.quantile(0.01)
        high = series.quantile(0.99)
        # the two is a complete hack
        inner_range = (high - low) / 2
        return series.between(low - inner_range, high + inner_range)
    mask = data.apply(_find_outliers_series)
    mask = mask.apply(lambda x: reduce(np.logical_and, x), axis=1).fillna(True)
    dropped = len(mask) - mask.sum()
    if dropped > 0:
        warn("Dropped {} outliers.".format(int(dropped)), UserWarning)
        return mask
    return None


def _get_scatter_alpha(scatter_alpha, x):
    if scatter_alpha != "auto":
        return scatter_alpha
    if x.shape[0] < 100:
        return .9
    elif x.shape[0] < 1000:
        return .5
    elif x.shape[0] < 10000:
        return .2
    else:
        return .1


def _get_scatter_size(scatter_size, x):
    if scatter_size != "auto":
        return scatter_size
    if x.shape[0] < 100:
        return 30
    elif x.shape[0] < 1000:
        return 30
    elif x.shape[0] < 2000:
        return 10
    elif x.shape[0] < 10000:
        return 2
    else:
        return 1


def plot_multiclass_roc_curve(estimator, X_val, y_val):
    if len(estimator.classes_) < 3:
        raise ValueError("Only for multi-class")
    try:
        y_score = estimator.predict_proba(X_val)
    except AttributeError:
        y_score = estimator.decision_function(X_val)
    fig, axes = _make_subplots(len(estimator.classes_))
    for i, (ax, c) in enumerate(zip(axes.ravel(), estimator.classes_)):
        fpr, tpr, _ = roc_curve(y_val == c, y_score[:, i])
        ax.plot(fpr, tpr)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate (recall)")
        ax.set_title("ROC curve for class {}".format(c))
