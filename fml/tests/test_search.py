import pytest

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from fml.search import GridSuccessiveHalving, RandomSuccessiveHalving


parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
base_estimator = SVC(gamma='scale')


# @pytest.mark.parametrize('sh', (
#     GridSuccessiveHalving(base_estimator, parameters),
#     RandomSuccessiveHalving(base_estimator, parameters, n_iter=4)
# ))
# def test_basic(sh):
#     X, y = make_classification(n_samples=1000, random_state=0)
#     sh.set_params(random_state=0, cv=5)
#     sh.fit(X, y)
#     assert sh.score(X, y) > .98



def test_lol():
    print('More than enough samples, exhaust_budget=True')
    parameters = {'penalty': ('l1', 'l2'), 'C': [1, 10, 20]}
    base_estimator = LogisticRegression(solver='liblinear')
    X, y = make_classification(n_samples=1000)
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
                               random_state=0, exhaust_budget=False, ratio=3)
    sh.fit(X, y)
    assert sh.n_iterations_ ==  2
    assert sh.n_required_iterations_ ==  2
    assert sh.n_possible_iterations_ > 2
    assert sh._r_i_list == [20, 60]

    print()
    print('-' * 10)
    print('-' * 10)
    print()

    print('More than enough samples, exhaust_budget=True')
    parameters = {'penalty': ('l1', 'l2'), 'C': [1, 10, 20]}
    base_estimator = LogisticRegression(solver='liblinear')
    X, y = make_classification(n_samples=1000)
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5, random_state=0,
                            exhaust_budget=True, ratio=3)
    sh.fit(X, y)
    assert sh.n_iterations_ ==  2
    assert sh.n_required_iterations_ ==  2
    assert sh.n_possible_iterations_ > 2
    assert sh._r_i_list == [333, 999]

    print()
    print('-' * 10)
    print('-' * 10)
    print()

    print('Not enough samples, finish before n_required_itearations')
    parameters = {'penalty': ('l1', 'l2'), 'C': list(range(1, 20))}
    base_estimator = LogisticRegression(solver='liblinear')
    X, y = make_classification(n_samples=1000)
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5, random_state=0,
                               aggressive_elimination=False, max_budget=76,
                               ratio=3)
    sh.fit(X, y)
    assert sh.n_iterations_ ==  2
    assert sh.n_required_iterations_ ==  4
    assert sh.n_possible_iterations_ == 2
    assert sh._r_i_list == [20, 60]

    print()
    print('-' * 10)
    print('-' * 10)
    print()

    print('Not enough samples (need to loop at the beginning), aggressive_elimination=True')
    parameters = {'penalty': ('l1', 'l2'), 'C': list(range(1, 50))}
    base_estimator = LogisticRegression(solver='liblinear')
    X, y = make_classification(n_samples=1000)
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5, random_state=0,
                               aggressive_elimination=True, max_budget=190,
                               ratio=3)
    sh.fit(X, y)

    assert sh.n_iterations_ ==  5
    assert sh.n_required_iterations_ ==  sh.n_iterations_
    assert sh.n_possible_iterations_ < sh.n_iterations_
    assert sh._r_i_list == [20, 20, 20, 60, 180]