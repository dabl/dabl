import re
import pandas as pd

from scipy.sparse import issparse
from inspect import signature
from sklearn.base import _pprint
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


def _changed_params(est):
    params = est.get_params(deep=False)
    filtered_params = {}
    init = getattr(est.__init__, 'deprecated_original', est.__init__)
    init_params = signature(init).parameters
    for k, v in params.items():
        if v != init_params[k].default and k != "random_state":
            if k == "multi_class" and v == "auto":
                # this is the new default
                continue
            filtered_params[k] = v
    return filtered_params


def nice_repr(est):
    class_name = est.__class__.__name__
    changed_params = _changed_params(est)
    name = ('%s(%s)' % (class_name, _pprint(changed_params,
                                            offset=len(class_name))))
    name, _ = re.subn(r"\s+", " ", name)
    return name
