from math import ceil, floor, log
from abc import abstractmethod
from copy import deepcopy

import numpy as np
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection._search import _check_param_grid
from sklearn.model_selection import ParameterGrid, ParameterSampler
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
                 force_exhaust_budget=False):

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
        self.force_exhaust_budget = force_exhaust_budget

    def _check_input_parameters(self, X, y, groups):

        if (self.budget_on != 'n_samples' and
                self.budget_on not in self.estimator.get_params()):
            raise ValueError(
                'Cannot budget on parameter {} which is not supported '
                'by estimator {}'.format(self.budget_on,
                                        self.estimator.__class__.__name__))

        if isinstance(self.max_budget, str) and self.max_budget != 'auto':
            raise ValueError(
                "max_budget must be either 'auto' or a positive number"
            )
        if self.max_budget != 'auto' and self.max_budget <= 0:
            raise ValueError(
                "max_budget must be either 'auto' or a positive number"
            )

        if isinstance(self.r_min, str) and self.r_min != 'auto':
            raise ValueError(
                "r_min must be either 'auto' or a positive number no greater "
                "than max_budget."
            )
        if self.r_min != 'auto' and self.r_min <= 0:
            raise ValueError(
                "r_min must be either 'auto' or a positive number no greater "
                "than max_budget."
            )

        if self.force_exhaust_budget and self.r_min != 'auto':
            raise ValueError(
                'r_min must be set to auto if force_exhaust_budget is True.'
            )

        self.r_min_ = self.r_min
        if self.r_min_ == 'auto':
            if self.budget_on == 'n_samples':
                cv = check_cv(self.cv, y,
                              classifier=is_classifier(self.estimator))
                n_splits = cv.get_n_splits(X, y, groups)

                # please see https://gph.is/1KjihQe for a justification
                magic_factor = 2
                self.r_min_ = n_splits * magic_factor
                if is_classifier(self.estimator):
                    n_classes = np.unique(y).shape[0]
                    self.r_min_ *= n_classes
            else:
                self.r_min_ = 1

        self.max_budget_ = self.max_budget
        if self.max_budget_ == 'auto':
            if self.budget_on == 'n_samples':
                self.max_budget_ = X.shape[0]
            else:
                self.max_budget_ = 20  # FIXME  # n_candidates * r_min??

        if self.r_min_ > self.max_budget_:
            raise ValueError(
                'r_min_={} is greater than max_budget_={}.'
                .format(self.r_min_, self.max_budget_)
            )

    def _run_search(self, evaluate_candidates, X, y, groups):

        self._check_input_parameters(
            X=X,
            y=y,
            groups=groups,
        )
        rng = check_random_state(self.random_state)

        candidate_params = list(self._generate_candidate_params())
        self.n_candidates_ = len(candidate_params)

        if self.budget_on != 'n_samples' and any(
                self.budget_on in candidate for candidate in candidate_params):
            # Can only check this now since we need the candidates list
            raise ValueError(
                "Cannot budget on parameter {} since it is part of "
                "the searched parameters.".format(self.budget_on))

        # n_required_iterations is the number of iterations needed so that the
        # last iterations evaluates less than `ratio` candidates.
        n_required_iterations = 1 + floor(log(self.n_candidates_, self.ratio))

        if self.force_exhaust_budget:
            # To exhaust the budget, we want to start with the biggest r_min
            # possible so that the last (required) iteration uses as many
            # resources as possible
            # We only force exhausting the budget if r_min wasn't specified by
            # the user.
            last_iteration = n_required_iterations - 1
            self.r_min_ = max(self.r_min_,
                              self.max_budget_ // self.ratio**last_iteration)

        # n_possible iterations is the number of iterations that we can
        # actually do starting from r_min and without exceeding the budget.
        # Depending on budget size the number of candidates, this may be higher
        # or smaller than n_required_iterations.
        n_possible_iterations = 1 + floor(log(self.max_budget_ // self.r_min_,
                                              self.ratio))

        if self.aggressive_elimination:
            n_iterations = n_required_iterations
        else:
            n_iterations = min(n_possible_iterations, n_required_iterations)

        if self.verbose:
            print(f'n_iterations: {n_iterations}')
            print(f'n_required_iterations: {n_required_iterations}')
            print(f'n_possible_iterations: {n_possible_iterations}')
            print(f'r_min_: {self.r_min_}')
            print(f'max_budget_: {self.max_budget_}')
            print(f'aggressive_elimination: {self.aggressive_elimination}')
            print(f'force_exhaust_budget: {self.force_exhaust_budget}')
            print(f'ratio: {self.ratio}')

        self._r_i_list = []  # list of r_i for each iteration, used in tests

        for iter_i in range(n_iterations):

            power = iter_i  # default
            if self.aggressive_elimination:
                # this will set r_i to the initial value (i.e. the value of r_i
                # at the first iteration) for as many iterations as needed
                # (while candidates are being eliminated), and then go on as
                # usual.
                power = max(
                    0,
                    iter_i - n_required_iterations + n_possible_iterations
                )

            r_i = int(self.ratio**power * self.r_min_)
            r_i = min(r_i, self.max_budget_)  # guard, probably not needed
            self._r_i_list.append(r_i)

            n_candidates = len(candidate_params)

            if self.verbose:
                print('-' * 10)
                print(f'iter_i: {iter_i}')
                print(f'n_candidates: {n_candidates}')
                print(f'r_i: {r_i}')
                print(f'r_i (in r_min units): {r_i // self.r_min_}')

            if self.budget_on == 'n_samples':
                # XXX FIXME TODO
                # subsampling should be stratified. We can't use
                # train_test_split because it complains about testset being too
                # small in some cases
                indexes = rng.choice(r_i, X.shape[0])
                X_iter, y_iter = X[indexes], y[indexes]
            else:
                candidate_params = deepcopy(candidate_params)
                for candidate in candidate_params:
                    candidate[self.budget_on] = r_i
                X_iter, y_iter = X, y

            more_results = {'iter': [iter_i] * n_candidates,
                            'r_i': [r_i] * n_candidates}
            results = evaluate_candidates(candidate_params, X_iter, y_iter,
                                          groups, more_results=more_results)

            n_candidates_to_keep = ceil(n_candidates / self.ratio)
            candidate_params = self._top_k(results,
                                           n_candidates_to_keep,
                                           iter_i)

        self.n_remaining_candidates_ = len(candidate_params)
        self.n_required_iterations_ = n_required_iterations
        self.n_possible_iterations_ = n_possible_iterations
        self.n_iterations_ = n_iterations

    def _top_k(self, results, k, iter_i):
        # Return the best candidates of a given iteration
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
                 force_exhaust_budget=False):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, verbose=verbose, cv=cv,
                         pre_dispatch=pre_dispatch,
                         random_state=random_state, error_score=error_score,
                         return_train_score=return_train_score,
                         max_budget=max_budget, budget_on=budget_on,
                         ratio=ratio, r_min=r_min,
                         aggressive_elimination=aggressive_elimination,
                         force_exhaust_budget=force_exhaust_budget)
        self.param_grid = param_grid
        _check_param_grid(self.param_grid)

    def _generate_candidate_params(self):
        return ParameterGrid(self.param_grid)


class RandomSuccessiveHalving(BaseSuccessiveHalving):

    def __init__(self, estimator, param_distributions, n_candidates='auto',
                 scoring=None, n_jobs=None, refit=True, verbose=0, cv=None,
                 pre_dispatch='2*n_jobs', random_state=None,
                 error_score=np.nan, return_train_score=True,
                 max_budget='auto', budget_on='n_samples', ratio=3,
                 r_min='auto', aggressive_elimination=False,
                 force_exhaust_budget=False):
        super().__init__(estimator, scoring=scoring,
                         n_jobs=n_jobs, verbose=verbose, cv=cv,
                         random_state=random_state, error_score=error_score,
                         return_train_score=return_train_score,
                         max_budget=max_budget, budget_on=budget_on,
                         ratio=ratio, r_min=r_min,
                         aggressive_elimination=aggressive_elimination,
                         force_exhaust_budget=force_exhaust_budget)
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates

    def _generate_candidate_params(self):
        n_candidates_ = self.n_candidates
        if n_candidates_ == 'auto':
            # This will generate enough candidate so that the last iteration
            # uses as much budget as possible
            n_candidates_ = self.max_budget_ // self.r_min_
        return ParameterSampler(self.param_distributions, n_candidates_,
                                self.random_state)
