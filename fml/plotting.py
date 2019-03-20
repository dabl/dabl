from .preprocessing import detect_types_dataframe

from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt


def plot_continuous_unsupervised(X):
    pass


def plot_categorical_unsupervised(X):
    pass


def _shortname(some_string):
    if len(some_string) > 20:
        return some_string[:17] + "..."
    else:
        return some_string

def plot_unsupervised(X, verbose=10):
    types = detect_types_dataframe(X)
    # if any dirty floats, tell user to clean them first
    n_types = types.sum()
    plot_continuous_unsupervised(X.loc[:, types.continous])
    plot_categorical_unsupervised(X.loc[:, types.categorical])


def plot_regression_continuous(X, target_col):
    if X.shape[1] > 20:
        print("Showing only top 10 of {} continuous features".format(X.shape[1]))
        # too many features, show just top 10
        show_top = 10
    else:
        show_top = X.shape[1]
    features = X.drop(target_col, axis=1)
    target = X[target_col]
    # HACK we should drop them per column before feeding them into f_regression
    # FIXME
    features_imp = SimpleImputer().fit_transform(features)
    f, p = f_regression(features_imp, target)
    top_k = np.argsort(f)[-show_top:][::-1]
    # we could do better lol
    n_cols = 5
    n_rows = int(np.ceil(show_top / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows),
                             constrained_layout=True)
    plt.suptitle("Continuous Target Plots")
    for i, (col, ax) in enumerate(zip(top_k, axes.ravel())):
        if i % n_cols == 0:
            ax.set_ylabel(target_col)
        ax.plot(features.iloc[:, col], target, 'o', alpha=.6)
        ax.set_xlabel(_shortname(features.columns[col]))
        ax.set_title("F={:.2E}".format(f[col]))

    for j in range(i + 1, n_rows * n_cols):
        # turn off axis if we didn't fill last row
        axes.ravel()[j].set_axis_off()


def plot_regression_categorical(X, target):
    pass


def plot_supervised(X, target, verbose=10):
    types = detect_types_dataframe(X)
    # if any dirty floats, tell user to clean them first
    if types.continuous[target]:
        print("regression")
        # regression
        plot_regression_continuous(X.loc[:, types.continuous], target)
        plot_regression_categorical(X.loc[:, types.categorical], target)
    else:
        print("classification")
        pass
        # classification
