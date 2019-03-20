from .preprocessing import detect_types_dataframe

from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt


def plot_continuous_unsupervised(X):
    pass


def plot_categorical_unsupervised(X):
    pass


def plot_unsupervised(X, verbose=10):
    types = detect_types_dataframe(X)
    # if any dirty floats, tell user to clean them first
    n_types = types.sum()
    plot_continuous_unsupervised(X.loc[:, types.continous])
    plot_categorical_unsupervised(X.loc[:, types.categorical])


def plot_regression_continuous(X, target):
    if X.shape[1] > 20:
        # too many features, show just top 10
        show_top = 10
    else:
        show_top = X.shape[1]
    features = X.drop(target, axis=1)
    target = X[target]
    # HACK we should drop them per column before feeding them into f_regression
    # FIXME
    features_imp = SimpleImputer().fit_transform(features)
    f, p = f_regression(features_imp, target)
    top_k = np.argsort(f)[-show_top:][::-1]
    # we could do better lol
    n_cols = 5
    fig, axes = plt.subplots(int(np.ceil(show_top / n_cols)), n_cols, figsize=(20, 5))
    for col, ax in zip(top_k, axes.ravel()):
        ax.plot(features.iloc[:, col], target, 'o', alpha=.6)
        ax.set_title("{}, {:.2f}".format(_shortname(features.columns[col]), f[col]))



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
