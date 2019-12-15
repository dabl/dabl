import pandas as pd
import numpy as np
from os.path import dirname, join


def load_ames():
    """Load ames housing dataset.

    Returns
    -------
    data : DataFrame
        DataFrame containing the ames housing dataset.
    """
    module_path = dirname(__file__)
    return pd.read_pickle(join(module_path, 'ames_housing.pkl.bz2'))


def load_titanic():
    """Load titanic dataset.

    Returns
    -------
    data : DataFrame
        DataFrame containing the titanic dataset.
    """
    module_path = dirname(__file__)
    return pd.read_csv(join(module_path, 'titanic.csv'))


def load_adult():
    """Load adult census dataset.

    Returns
    -------
    data : DataFrame
        DataFrame containing the adult dataset.
    """
    module_path = dirname(__file__)
    return pd.read_csv(join(module_path, 'adult.csv.gz'))


def data_path(filename):
    module_path = dirname(__file__)
    return join(module_path, filename)


def make_multiclass_strawman():
    X = np.random.normal(0, .1, size=[500, 6])
    y = np.arange(500) // 100

    X[100:200, 0] += 2
    X[200:300, 1] += 2

    X[300:400, 0] += 2
    X[300:400, 1] += 2

    X[400: 500, 2] += 1

    X[:, 3] = X[:, 0]
    X[:, 4] = X[:, 1]
    X[:, 5] = X[:, 1]
    return X, y
