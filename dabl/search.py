from sklearn.experimental import enable_halving_search_cv
from sklearn.utils import deprecated
from sklearn.model_selection import HalvingGridSearchCV  as HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV  as HalvingRandomSearchCV


__all__ = ['GridSuccessiveHalving', 'RandomSuccessiveHalving']


@deprecated("GridSuccessiveHalving was upstreamed to sklearn,"
            " please import from sklearn.model_selection.")
class GridSuccessiveHalving(HalvingGridSearchCV):
    """Grid-search with successive halving.

    The search strategy for hyper-parameter optimization starts evaluating all
    the candidates with a small amount of resource and iteratively selects the
    best candidates, using more and more resources.

    Read more in the :ref:`User guide<successive_halving_user_guide>`.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (string) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : string, callable, or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        If None, the estimator's score method is used.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    cv : int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer to :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        If True, refit an estimator using the best found parameters on the
        whole dataset.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``

    return_train_score : boolean, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    max_budget : int, optional(default='auto')
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. By default, this is set ``n_samples`` when
        ``budget_on='n_samples'`` (default), else an error is raised.

    budget_on : `n_samples` or str, optional(default='n_samples')
        Defines the nature of the budget. By default, the budget is the number
        of samples. It can also be set to any parameter of the base estimator
        that accepts positive integer values, e.g. 'n_iterations' or
        'n_estimators' for a gradient boosting estimator. In this case
        ``max_budget`` cannot be 'auto'.

    ratio : int or float, optional(default=3)
        The 'halving' parameter, which determines the proportion of candidates
        that are selected for the next iteration. For example, ``ratio=3``
        means that only one third of the candidates are selected.

    r_min : int, optional(default='auto')
        The minimum amount of resource that any candidate is allowed to use for
        a given iteration. Equivalently, this defines the amount of resources
        that are allocated for each candidate at the first iteration. By
        default, this is set to:

        - ``n_splits * 2`` when ``budget_on='n_samples'`` for a regression
          problem
        - ``n_classes * n_splits * 2`` when ``budget_on='n_samples'`` for a
          regression problem
        - The highest possible value satisfying the constraint
          ``force_exhaust_budget=True``.
        - ``1`` when ``budget_on!='n_samples'``

        Note that the amount of resources used at each iteration is always a
        multiple of ``r_min``.

    aggressive_elimination : bool, optional(default=False)
        This is only relevant in cases where there isn't enough budget to
        eliminate enough candidates at the last iteration. If ``True``, then
        the search process will 'replay' the first iteration for as long as
        needed until the number of candidates is small enough. This is
        ``False`` by default, which means that the last iteration may evaluate
        more than ``ratio`` candidates.

    force_exhaust_budget : bool, optional(default=False)
        If True, then ``r_min`` is set to a specific value such that the
        last iteration uses as much budget as possible. Namely, the last
        iteration uses the highest value smaller than ``max_budget`` that is a
        multiple of both ``r_min`` and ``ratio``.

    Attributes
    ----------
    n_candidates_ : int
        The number of candidate parameters that were evaluated at the first
        iteration.

    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration.

    max_budget_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used at
        each iteration must be a multiple of ``r_min_``, the actual number of
        resources used at the last iteration may be smaller than
        ``max_budget_``.

    r_min_ : int
        The amount of resources that are allocated for each candidate at the
        first iteration.

    n_iterations_ : int
        The actual number of iterations that were run. This is equal to
        ``n_required_iterations_`` if ``aggressive_elimination`` is ``True``.
        Else, this is equal to ``min(n_possible_iterations_,
        n_required_iterations_)``.

    n_possible_iterations_ : int
        The number of iterations that are possible starting with ``r_min_``
        resources and without exceeding ``max_budget_``.

    n_required_iterations_ : int
        The number of iterations that are required to end up with less than
        ``ratio`` candidates at the last iteration, starting with ``r_min_``
        resources. This will be smaller than ``n_possible_iterations_`` when
        there isn't enough budget.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting (and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`RandomSuccessiveHalving`:
        Random search over a set of parameters using successive halving.
    """
    pass

@deprecated("RandomSuccessiveHalving was upstreamed to sklearn,"
            " please import from sklearn.model_selection.")
class RandomSuccessiveHalving(HalvingRandomSearchCV):
    """Randomized search with successive halving.

    The search strategy for hyper-parameter optimization starts evaluating all
    the candidates with a small amount a resource and iteratively selects the
    best candidates, using more and more resources.

    Read more in the :ref:`User guide<successive_halving_user_guide>`.

    Parameters
    ----------
    estimator : estimator object.
        This is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_distributions : dict
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.

    n_candidates: int, optional(default='auto')
        The number of candidate parameters to sample. By default this will
        sample enough candidates so that the last iteration uses as many
        resources as possible. Note that ``force_exhaust_budget`` has no
        effect in this case.

    scoring : string, callable, or None, default: None
        A single string (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.
        If None, the estimator's score method is used.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    cv : int, cross-validation generator or an iterable, optional (default=5)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - integer, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer to :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    refit : boolean, default=True
        If True, refit an estimator using the best found parameters on the
        whole dataset.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        ``GridSearchCV`` instance.

    verbose : integer
        Controls the verbosity: the higher, the more messages.

    error_score : 'raise' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error. Default is ``np.nan``

    return_train_score : boolean, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    max_budget : int, optional(default='auto')
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. By default, this is set ``n_samples`` when
        ``budget_on='n_samples'`` (default), else an error is raised.

    budget_on : `n_samples` or str, optional(default='n_samples')
        Defines the nature of the budget. By default, the budget is the number
        of samples. It can also be set to any parameter of the base estimator
        that accepts positive integer values, e.g. 'n_iterations' or
        'n_estimators' for a gradient boosting estimator. In this case
        ``max_budget`` cannot be 'auto'.

    ratio : int or float, optional(default=3)
        The 'halving' parameter, which determines the proportion of candidates
        that are selected for the next iteration. For example, ``ratio=3``
        means that only one third of the candidates are selected.

    r_min : int, optional(default='auto')
        The minimum amount of resource that any candidate is allowed to use for
        a given iteration. Equivalently, this defines the amount of resources
        that are allocated for each candidate at the first iteration. By
        default, this is set to:

        - ``n_splits * 2`` when ``budget_on='n_samples'`` for a regression
          problem
        - ``n_classes * n_splits * 2`` when ``budget_on='n_samples'`` for a
          regression problem
        - The highest possible value satisfying the constraint
          ``force_exhaust_budget=True``.
        - ``1`` when ``budget_on!='n_samples'``

        Note that the amount of resources used at each iteration is always a
        multiple of ``r_min``.

    aggressive_elimination : bool, optional(default=False)
        This is only relevant in cases where there isn't enough budget to
        eliminate enough candidates at the last iteration. If ``True``, then
        the search process will 'replay' the first iteration for as long as
        needed until the number of candidates is small enough. This is
        ``False`` by default, which means that the last iteration may evaluate
        more than ``ratio`` candidates.

    force_exhaust_budget : bool, optional(default=False)
        If True, then ``r_min`` is set to a specific value such that the
        last iteration uses as much budget as possible. Namely, the last
        iteration uses the highest value smaller than ``max_budget`` that is a
        multiple of both ``r_min`` and ``ratio``.

    Attributes
    ----------
    n_candidates_ : int
        The number of candidate parameters that were evaluated at the first
        iteration.

    n_remaining_candidates_ : int
        The number of candidate parameters that are left after the last
        iteration.

    max_budget_ : int
        The maximum number of resources that any candidate is allowed to use
        for a given iteration. Note that since the number of resources used at
        each iteration must be a multiple of ``r_min_``, the actual number of
        resources used at the last iteration may be smaller than
        ``max_budget_``.

    r_min_ : int
        The amount of resources that are allocated for each candidate at the
        first iteration.

    n_iterations_ : int
        The actual number of iterations that were run. This is equal to
        ``n_required_iterations_`` if ``aggressive_elimination`` is ``True``.
        Else, this is equal to ``min(n_possible_iterations_,
        n_required_iterations_)``.

    n_possible_iterations_ : int
        The number of iterations that are possible starting with ``r_min_``
        resources and without exceeding ``max_budget_``.

    n_required_iterations_ : int
        The number of iterations that are required to end up with less than
        ``ratio`` candidates at the last iteration, starting with ``r_min_``
        resources. This will be smaller than ``n_possible_iterations_`` when
        there isn't enough budget.

    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For instance the below given table

        +--------------+-------------+-------------------+---+---------------+
        | param_kernel | param_gamma | split0_test_score |...|rank_test_score|
        +==============+=============+===================+===+===============+
        |    'rbf'     |     0.1     |       0.80        |...|       2       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.2     |       0.90        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+
        |    'rbf'     |     0.3     |       0.70        |...|       1       |
        +--------------+-------------+-------------------+---+---------------+

        will be represented by a ``cv_results_`` dict of::

            {
            'param_kernel' : masked_array(data = ['rbf', 'rbf', 'rbf'],
                                          mask = False),
            'param_gamma'  : masked_array(data = [0.1 0.2 0.3], mask = False),
            'split0_test_score'  : [0.80, 0.90, 0.70],
            'split1_test_score'  : [0.82, 0.50, 0.70],
            'mean_test_score'    : [0.81, 0.70, 0.70],
            'std_test_score'     : [0.01, 0.20, 0.00],
            'rank_test_score'    : [3, 1, 1],
            'split0_train_score' : [0.80, 0.92, 0.70],
            'split1_train_score' : [0.82, 0.55, 0.70],
            'mean_train_score'   : [0.81, 0.74, 0.70],
            'std_train_score'    : [0.01, 0.19, 0.00],
            'mean_fit_time'      : [0.73, 0.63, 0.43],
            'std_fit_time'       : [0.01, 0.02, 0.01],
            'mean_score_time'    : [0.01, 0.06, 0.04],
            'std_score_time'     : [0.00, 0.00, 0.00],
            'params'             : [{'kernel' : 'rbf', 'gamma' : 0.1}, ...],
            }

        NOTE

        The key ``'params'`` is used to store a list of parameter
        settings dicts for all the parameter candidates.

        The ``mean_fit_time``, ``std_fit_time``, ``mean_score_time`` and
        ``std_score_time`` are all in seconds.

    best_estimator_ : estimator or dict
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

    best_score_ : float
        Mean cross-validated score of the best_estimator.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    Notes
    -----
    The parameters selected are those that maximize the score of the held-out
    data, according to the scoring parameter.

    If `n_jobs` was set to a value higher than one, the data is copied for each
    parameter setting(and not `n_jobs` times). This is done for efficiency
    reasons if individual jobs take very little time, but may raise errors if
    the dataset is large and not enough memory is available.  A workaround in
    this case is to set `pre_dispatch`. Then, the memory is copied only
    `pre_dispatch` many times. A reasonable value for `pre_dispatch` is `2 *
    n_jobs`.

    See Also
    --------
    :class:`GridSuccessiveHalving`:
        Search over a grid of parameters using successive halving.
    """
    pass