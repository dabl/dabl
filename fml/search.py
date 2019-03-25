from math import ceil, floor, log
from abc import abstractmethod

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._search import _check_param_grid
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.base import is_classifier
from sklearn.model_selection._split import check_cv


__all__ = ['GridSuccessiveHalving', 'RandomSuccessiveHalving']


def _refit_callable(results):
    # Custom refit callable to return the index of the best candidate. We want
    # the best candidate out of the last iteration. By default BaseSearchCV
    # would return the best candidate out of all iterations.

    last_iter = np.max(results['iter'])
    sorted_indices = np.argsort(results['mean_test_score'])[::-1]
    best_index = next(i for i in sorted_indices
                      if results['iter'][i] == last_iter)
    return best_index


class BaseSuccessiveHalving(BaseSearchCV):
    """Implements successive halving.

    Ref:
    Almost optimal exploration in multi-armed bandits, ICML 13
    Zohar Karnin, Tomer Koren, Oren Somekh
    """
    def __init__(self, estimator, scoring=None,
                 n_jobs=None, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=True,
                 max_budget='auto', budget_on='n_samples', ratio=3,
                 r_min='auto', aggressive_elimination=False,
                 exhaust_budget=False):

        refit = _refit_callable if refit else False
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, refit=refit, cv=cv,
                         verbose=verbose, pre_dispatch=pre_dispatch,
                         error_score=error_score,
                         return_train_score=return_train_score, iid=False)

        self.random_state = random_state
        self.max_budget = max_budget
        self.budget_on = budget_on
        self.ratio = ratio
        self.r_min = r_min
        self.aggressive_elimination = aggressive_elimination
        self.exhaust_budget = exhaust_budget

    def _check_input_parameters(self, X, y, n_candidates):

        if self.budget_on != 'n_samples':
            raise ValueError('budget_on must be n_samples for now')

        self.r_min_ = self.r_min
        if self.r_min_ == 'auto':
            if self.budget_on == 'n_samples':
                cv = check_cv(self.cv, y,
                              classifier=is_classifier(self.estimator))
                n_splits = cv.get_n_splits(X, y)  # FIXME: should pass group :/

                magic_factor = 2
                self.r_min_ = n_splits * magic_factor
                if is_classifier(self.estimator):
                    n_classes = np.unique(y).shape[0]
                    self.r_min_ *= n_classes
            else:
                self.r_min_ = 1

        self.max_budget_ = self.max_budget
        if self.max_budget_ == 'auto':
            self.max_budget_ = X.shape[0]  # only budget_on=n_samples allowed

        # We want resources to be in units of r_min_
        self.max_budget_ = self.max_budget_ // self.r_min_

    def _run_search(self, evaluate_candidates, X, y):

        candidate_params = list(self._generate_candidate_params())
        n_candidates = len(candidate_params)

        self._check_input_parameters(
            X=X,
            y=y,
            n_candidates=n_candidates
        )
        rng = check_random_state(self.random_state)

        # n_iterations_required is the number of iterations needed so that the
        # last iterations evaluates less than `ratio` candidates.
        n_required_iterations = floor(log(n_candidates, self.ratio)) + 1

        actual_max_resources = self.max_budget_ * self.r_min_
        last_iter = n_required_iterations - 1
        r_last_iter = self.ratio**last_iter * self.r_min_
        if self.exhaust_budget:
            # To exhaust we want to start with the biggest
            # r_min possible so that the last iteration uses as much resources
            # as possible
            self.r_min_ = actual_max_resources // self.ratio**last_iter


        # n_possible iterations is the number of iterations that we can
        # actually do starting from r_min and without exceeding the budget.
        n_possible_iterations = floor(log(self.max_budget_, self.ratio)) + 1

        print(f'n_required_iterations: {n_required_iterations}')
        print(f'n_possible_iterations: {n_possible_iterations}')
        print(f'r_min: {self.r_min_}')
        print(f'max_budget_: {self.max_budget_}')
        print(f'aggressive_elimination: {self.aggressive_elimination}')
        print(f'ratio: {self.ratio}')

        self._r_i_list = []


        if self.aggressive_elimination:
            n_iterations = n_required_iterations
        else:
            n_iterations = min(n_possible_iterations, n_required_iterations)

        for iter_i in range(n_iterations):

            power = iter_i  # default
            if self.aggressive_elimination:
                power = max(0, iter_i - n_required_iterations + n_possible_iterations)

            r_i = int(self.ratio**power * self.r_min_)
            r_i = min(r_i, self.max_budget_ * self.r_min_)
            self._r_i_list.append(r_i)

            n_candidates = len(candidate_params)
            print('-' * 10)
            print(f'iter_i: {iter_i}')
            print(f'n_candidates: {n_candidates}')
            print(f'r_i: {r_i}')
            print(f'r_i (in r_min units): {r_i // self.r_min_}')

            if self.budget_on == 'n_samples':
                # ideally use train_test_split, but doesn't work lololollol
                # stratify = y if is_classifier(self.estimator) else None
                # X_iter, _, y_iter, _ = train_test_split(
                #     X, y, stratify=stratify, train_size=r_i, test_size=None,
                #     random_state=rng)
                indexes = rng.choice(r_i, X.shape[0])
                X_iter, y_iter = X[indexes], y[indexes]
            else:
                raise ValueError("I TOLD YOU NOT TO")

            more_results= {'iter': [iter_i] * n_candidates,
                           'r_i': [r_i] * n_candidates}
            results = evaluate_candidates(candidate_params, X_iter, y_iter,
                                          more_results=more_results)

            n_candidates_to_keep = ceil(n_candidates/ self.ratio)
            candidate_params = self._keep_k_best(results,
                                                 n_candidates_to_keep,
                                                 iter_i)

        self.n_required_iterations_ = n_required_iterations
        self.n_possible_iterations_ = n_possible_iterations
        self.n_iterations_ = n_iterations


    def _keep_k_best(self, results, k, iter_i):
        # Select the best candidates of a given iteration
        # We need to filter out candidates from the previous iterations
        # when sorting

        best_candidates_indices = np.argsort(results['mean_test_score'])[::-1]
        best_candidates_indices = [idx for idx in best_candidates_indices
                                   if results['iter'][idx] == iter_i]
        best_candidates_indices = best_candidates_indices[:k]
        return [results['params'][idx] for idx in best_candidates_indices]

    @abstractmethod
    def _generate_candidate_params(self):
        pass


class GridSuccessiveHalving(BaseSuccessiveHalving):

    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=None, refit=True, verbose=0, cv=None,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=True,
                 max_budget='auto', budget_on='n_samples', ratio=3,
                 r_min='auto', aggressive_elimination=False,
                 exhaust_budget=False):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, verbose=verbose, cv=cv,
                         pre_dispatch=pre_dispatch,
                         random_state=random_state, error_score=error_score,
                         return_train_score=return_train_score,
                         max_budget=max_budget, budget_on=budget_on,
                         ratio=ratio, r_min=r_min,
                         aggressive_elimination=aggressive_elimination,
                         exhaust_budget=exhaust_budget)
        self.param_grid = param_grid
        _check_param_grid(self.param_grid)

    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)


class RandomSuccessiveHalving(BaseSuccessiveHalving):

    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None,
                 n_jobs=None, refit=True, verbose=0, cv=None,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=True,
                 max_budget='auto', budget_on='n_samples', ratio=3,
                 r_min='auto', aggressive_elimination=False,
                 exhaust_budget=False):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, verbose=verbose, cv=cv,
                         random_state=random_state, error_score=error_score,
                         return_train_score=return_train_score,
                         max_budget=max_budget, budget_on=budget_on,
                         ratio=ratio, r_min=r_min,
                         aggressive_elimination=aggressive_elimination,
                         exhaust_budget=exhaust_budget)
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def _generate_candidate_params(self):
        return ParameterSampler(self.param_distributions, self.n_iter,
                                self.random_state)
