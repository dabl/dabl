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
    """Load ames housing dataset.

    Returns
    -------
    data : DataFrame
        DataFrame containing the ames housing dataset.
    """
    module_path = dirname(__file__)
    return pd.read_pickle(join(module_path, 'ames.pkl.bz2'))
