"""Global configuration. Not threadsafe, not copied by joblib."""

_global_config = {'truncate_labels': True}


def get_config():
    """Retrieve global configuration values set by set_config.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to set_config.

    See Also
    --------
    set_config : set global dabl configuration.
    """
    return _global_config.copy()


def set_config(truncate_labels=None):
    """Set global dabl configuration.

    Parameters
    ----------
    truncate_labels : bool, default=None
        Whether to truncate labels in plots.
    """
    if truncate_labels is not None:
        _global_config['truncate_labels'] = truncate_labels
