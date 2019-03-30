import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.dummy import DummyClassifier
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score


from ..preprocessing import detect_types


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


def plot_coefficients(coefficients, feature_names, n_top_features=10, classname=None):
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
        raise ValueError("coeffients must be 1d array or column vector, got"
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
    plt.figure(figsize=(len(interesting_coefficients), 5))
    colors = ['red' if c < 0 else 'blue'
              for c in coef[interesting_coefficients]]
    plt.bar(np.arange(len(interesting_coefficients)),
            coef[interesting_coefficients],
            color=colors)
    feature_names = np.array(feature_names)
    plt.subplots_adjust(bottom=0.3)
    plt.xticks(np.arange(0, len(interesting_coefficients)),
               feature_names[interesting_coefficients], rotation=60,
               ha="right")
    plt.ylabel("Coefficient magnitude")
    plt.xlabel("Feature")
    plt.title(classname)


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

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
    res = res.cat.add_categories(['dabl_other']).fillna("dabl_other")
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
            X[col].fillna("dabl_missing", inplace=True)
        else:
            X[col].fillna(max_value + 1, inplace=True)
    return X


def _make_subplots(n_plots, max_cols=5, row_height=3):
    n_rows, n_cols = find_pretty_grid(n_plots, max_cols=max_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, row_height * n_rows),
                             constrained_layout=True)
    # we don't want ravel to fail, this is awkward!
    axes = np.atleast_2d(axes)
    return fig, axes


def _check_X_target_col(X, target_col, types, task=None):
    if types is None:
        types = detect_types(X)
    if not isinstance(target_col, str) and len(target_col) > 1:
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