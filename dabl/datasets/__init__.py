import pandas as pd
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
