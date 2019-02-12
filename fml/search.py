from math import ceil, floor, log2
from abc import abstractmethod

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._search import _check_param_grid
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils import check_random_state


__all__ = ['GridSuccessiveHalving', 'RandomSuccessiveHalving']


def _refit_callable(results):
    # Return the best index from the results dict (gs.cv_results_). We need a
    # custom refit because by default best_index is computed accross all
    # candidates but here we only want the best candidate from the last
    # iteration.
    # Note: we hardcode that the last iteration has only 2 candidates here.
    # That might not be the case if we stop the search before the end.

    best_index_last_iteration = np.argmax(results['mean_test_score'][-2:])
    out = len(results['mean_test_score']) - 2 + best_index_last_iteration
    return out


class BaseSuccessiveHalving(BaseSearchCV):
    """Implements successive halving.

    Ref:
    Almost optimal exploration in multi-armed bandits, ICML 13
    Zohar Karnin, Tomer Koren, Oren Somekh


    Attributes
    ----------
    best_estimator_ : estimator
        The last estimator remaining after the halving procedure.
    best_params : dict
        The parameters of the best estimator.
    """
    def __init__(self, estimator, scoring,
                 n_jobs=None, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=True):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, refit=_refit_callable, cv=cv,
                         verbose=verbose, pre_dispatch=pre_dispatch,
                         error_score=error_score,
                         return_train_score=return_train_score, iid=False)

        self.random_state = random_state

    def _run_search(self, evaluate_candidates, X, y):
        rng = check_random_state(self.random_state)

        candidate_params = list(self._generate_candidate_params())
        n_iterations = int(ceil(log2(len(candidate_params))))
        n_samples_total = X.shape[0]

        for _ in range(n_iterations):
            # randomly sample training samples
            n_candidates = len(candidate_params)
            n_samples_iter = floor(n_samples_total /
                                   (n_candidates * n_iterations))
            indices = rng.choice(n_samples_total, n_samples_iter,
                                 replace=False)
            X_iter, y_iter = X[indices], y[indices]

            out = evaluate_candidates(candidate_params, X_iter, y_iter)

            # Select the best half of the candidates for the next iteration
            n_candidates_to_keep = ceil(n_candidates / 2)
            best_candidates_indices = np.argsort(out['mean_test_score'])
            best_candidates_indices = \
                best_candidates_indices[-n_candidates_to_keep:]
            candidate_params = [candidate_params[i]
                                for i in best_candidates_indices]

        assert len(candidate_params) == n_candidates_to_keep == 1

    @abstractmethod
    def _generate_candidate_params():
        pass


class GridSuccessiveHalving(BaseSuccessiveHalving):

    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=None, verbose=0, cv=None,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=True):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, verbose=verbose, cv=cv,
                         pre_dispatch=pre_dispatch,
                         random_state=random_state, error_score=error_score,
                         return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(self.param_grid)

    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)


class RandomSuccessiveHalving(BaseSuccessiveHalving):

    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 n_jobs=None, verbose=0, cv=None,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=True):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, verbose=verbose, cv=cv,
                         random_state=random_state, error_score=error_score,
                         return_train_score=return_train_score)
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def _generate_candidate_params(self):
        return ParameterSampler(self.param_distributions, self.n_iter,
                                self.random_state)
