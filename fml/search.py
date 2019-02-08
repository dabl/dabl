from math import ceil, floor, log2
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._search import _check_param_grid
from sklearn.model_selection._search import _fit_and_score
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils.validation import indexable
from sklearn.metrics.scorer import _check_multimetric_scoring
from sklearn.utils import shuffle
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.base import is_classifier


__all__ = ['GridSuccessiveHalving', 'RandomSuccessiveHalving']


class BaseSuccessiveHalving(BaseSearchCV):
    """Implements successive halving.

    Ref:
    Almost optimal exploration in multi-armed bandits, ICML 13
    Zohar Karnin, Tomer Koren, Oren Somekh
    """
    def __init__(self, estimator, scoring=None,
                 n_jobs=None, refit=True, verbose=0, cv=None,
                 pre_dispatch='2*n_jobs'):

        self.estimator = estimator
        self.scoring = scoring  # FIXME: support this?
        if scoring is not None:
            raise NotImplementedError
        self.n_jobs = n_jobs
        self.refit = refit  # FIXME
        if refit is not True:
            raise NotImplementedError
        self.verbose = verbose
        self.cv = cv
        self.pre_dispatch = pre_dispatch

    def fit(self, X, y, groups=None, **fit_params):

        X, y, groups = indexable(X, y, groups)
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)
        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=False,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    error_score=np.nan,
                                    verbose=self.verbose)

        candidate_params = list(self._generate_candidate_params())
        n_iterations = int(ceil(log2(len(candidate_params))))
        n_samples_total = X.shape[0]

        for _ in range(n_iterations):
            n_candidates = len(candidate_params)
            n_samples = floor(n_samples_total / (n_candidates * n_iterations))
            # FIXME: use shuffle split instead?
            shuffle(X, y)
            X_iter, y_iter = X[:n_samples], y[:n_samples]

            out = parallel(delayed(_fit_and_score)(clone(base_estimator),
                                                   X, y,
                                                   train=train, test=test,
                                                   parameters=parameters,
                                                   **fit_and_score_kwargs)
                           for parameters, (train, test)
                           in product(candidate_params,
                                      cv.split(X_iter, y_iter)))

            test_score_dicts = [res[0] for res in out]
            test_scores = _aggregate_score_dicts(test_score_dicts)['score']
            test_scores = test_scores.reshape(n_candidates, n_splits)
            avg_test_scores = test_scores.mean(axis=1)

            # Select the best half of the candidates for the next iteration
            n_candidates_to_keep = ceil(n_candidates / 2)
            best_candidates_indices = \
                np.argsort(avg_test_scores)[-n_candidates_to_keep:]
            candidate_params = [candidate_params[i]
                                for i in best_candidates_indices]

        # There's only one parameter combination left
        assert len(candidate_params) == 1
        self.best_params_ = candidate_params[0]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            self.best_estimator_.fit(X, y, **fit_params)
        self.scorer_ = scorers['score']


class GridSuccessiveHalving(BaseSuccessiveHalving):

    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=None, refit=True, verbose=0, cv=None,
                 pre_dispatch='2*n_jobs'):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, refit=refit, verbose=verbose, cv=cv,
                         pre_dispatch=pre_dispatch)
        self.param_grid = param_grid
        _check_param_grid(self.param_grid)

    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)


class RandomSuccessiveHalving(BaseSuccessiveHalving):

    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 n_jobs=None, refit=True, verbose=0, cv=None,
                 pre_dispatch='2*n_jobs', random_state=None):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, refit=refit, verbose=verbose, cv=cv,
                         pre_dispatch=pre_dispatch)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _generate_candidate_params(self):
        return ParameterSampler(self.param_distributions, self.n_iter,
                                self.random_state)