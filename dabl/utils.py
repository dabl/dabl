import pandas as pd

from scipy.sparse import issparse
from .preprocessing import clean


def data_df_from_bunch(data_bunch):
    try:
        feature_names = data_bunch.feature_names
    except AttributeError:
        feature_names = ['x%d' % i for i in range(data_bunch.data.shape[1])]
    data = data_bunch.data
    if issparse(data):
        data = data.toarray()
    df = pd.DataFrame(data, columns=feature_names)
    try:
        df['target'] = data_bunch.target_names[data_bunch.target]
    except (TypeError, AttributeError):
        df['target'] = data_bunch.target
    return df


def _validate_Xyt(X, y, target_col, do_clean=True):
    """Ensure y and target_col are exclusive.

    Make sure either y or target_col is passed.
    If target_col is passed, extract it from X.
    """
    if ((y is None and target_col is None)
            or (y is not None) and (target_col is not None)):
        raise ValueError(
            "Need to specify either y or target_col.")
    if do_clean:
        X = clean(X)
    elif not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if target_col is not None:
        y = X[target_col]
        X = X.drop(target_col, axis=1)
    elif not isinstance(y, (pd.Series, pd.DataFrame)):
        y = pd.Series(y)
    return X, y
