import pandas as pd

from inspect import signature
from sklearn.base import _pprint


def data_df_from_bunch(data_bunch):
    try:
        feature_names = data_bunch.feature_names
    except AttributeError:
        feature_names = ['x%d' % i for i in range(data_bunch.data.shape[1])]
    df = pd.DataFrame(data_bunch.data, columns=feature_names)
    try:
        df['target'] = data_bunch.target_names[data_bunch.target]
    except AttributeError:
        df['target'] = data_bunch.target
    return df


def _changed_params(est):
    params = est.get_params(deep=False)
    filtered_params = {}
    init = getattr(est.__init__, 'deprecated_original', est.__init__)
    init_params = signature(init).parameters
    for k, v in params.items():
        if v != init_params[k].default:
            filtered_params[k] = v
    return filtered_params


def nice_repr(est):
    class_name = est.__class__.__name__
    return ('%s(%s)' % (class_name, _pprint(_changed_params(est),
                                            offset=len(class_name))))
