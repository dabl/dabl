import pytest

from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

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

class FastClassifier(DummyClassifier):
    """Dummy classifier that accepts parameters a, b, ... z.

    These parameter don't affect the predictions and are useful for fast
    grid searching."""

    def __init__(self, strategy='stratified', random_state=None,
                 constant=None, **kwargs):
        super().__init__(strategy=strategy, random_state=random_state,
                         constant=constant)

    def get_params(self, deep=False):
        params = super().get_params(deep=deep)
        for char in range(ord('a'), ord('z') + 1):
            params[chr(char)] = 'whatever'
        return params

def test_aggressive_elimination():
    # Test the aggressive_elimination parameter.

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    parameters = {'a': ('l1', 'l2'), 'b': list(range(1, 30))}
    base_estimator = FastClassifier()
    ratio = 3

    # aggressive_elimination is only really relevant when there is not enough
    # budget.
    max_budget = 180

    # aggressive_elimination=True
    # In this case, the first iterations only use r_min_ resources
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
        aggressive_elimination=True, max_budget=max_budget, ratio=ratio)
    sh.fit(X, y)

    assert sh.n_iterations_ ==  4
    assert sh.n_required_iterations_ == 4
    assert sh.n_possible_iterations_ == 3
    assert sh._r_i_list == [20, 20, 60, 180]  # see how it loops at the start
    assert len(sh.remaining_candidates_) == 1

    # aggressive_elimination=False
    # In this case we don't loop at the start, and might end up with a lot of
    # candidates at the last iteration
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
        aggressive_elimination=False, max_budget=max_budget, ratio=ratio)
    sh.fit(X, y)

    assert sh.n_iterations_ ==  3
    assert sh.n_required_iterations_ == 4
    assert sh.n_possible_iterations_ == 3
    assert sh._r_i_list == [20, 60, 180]
    assert len(sh.remaining_candidates_) == 3

    max_budget = n_samples
    # with enough budget, aggressive_elimination has no effect since it is not
    # needed

    # aggressive_elimination=True
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
        aggressive_elimination=True, max_budget=max_budget, ratio=ratio)
    sh.fit(X, y)

    assert sh.n_iterations_ ==  4
    assert sh.n_required_iterations_ == 4
    assert sh.n_possible_iterations_ == 4
    assert sh._r_i_list == [20, 60, 180, 540]
    assert len(sh.remaining_candidates_) == 1

    # aggressive_elimination=False
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
        aggressive_elimination=False, max_budget=max_budget, ratio=ratio)
    sh.fit(X, y)

    assert sh.n_iterations_ ==  4
    assert sh.n_required_iterations_ == 4
    assert sh.n_possible_iterations_ == 4
    assert sh._r_i_list == [20, 60, 180, 540]
    assert len(sh.remaining_candidates_) == 1


def test_exhaust_budget_false():
    # Test the exhaust_budget parameter when it's false or ignored.
    # This is the default case: we start at the beginning no matter what since
    # we do not overwrite r_min_

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    parameters = {'a': [1, 2], 'b': [1, 2, 3]}
    base_estimator = FastClassifier()
    ratio = 3

    # with enough budget
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
                               exhaust_budget=False, ratio=ratio)
    sh.fit(X, y)
    assert sh.n_iterations_ ==  2
    assert sh.n_required_iterations_ ==  2
    assert sh.n_possible_iterations_ == 4
    assert sh._r_i_list == [20, 60]

    # with enough budget but r_min!='auto': ignored
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
                               exhaust_budget=False, ratio=ratio,
                               r_min=50)
    sh.fit(X, y)
    assert sh.n_iterations_ ==  2
    assert sh.n_required_iterations_ ==  2
    assert sh.n_possible_iterations_ == 3
    assert sh._r_i_list == [50, 150]

    # without enough budget (budget is exhausted anyway)
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
                               exhaust_budget=False, ratio=ratio,
                               max_budget=30)
    sh.fit(X, y)
    assert sh.n_iterations_ ==  1
    assert sh.n_required_iterations_ ==  2
    assert sh.n_possible_iterations_ == 1
    assert sh._r_i_list == [20]

@pytest.mark.parametrize(
    'max_budget, r_i_list', [
    ('auto', [333, 999]),
    (1000, [333, 999]),
    (999, [333, 999]),
    (600, [200, 600]),
    (599, [199, 597]),
    (300, [100, 300]),
    (60, [20, 60]),
    (50, [20]),
    (20, [20]),
])
def test_exhaust_budget_true(max_budget, r_i_list):
    # Test the exhaust_budget parameter when it's true
    # in this case we need to change r_min so that the last iteration uses as
    # much budget as possible

    n_samples = 1000
    X, y = make_classification(n_samples=n_samples, random_state=0)
    parameters = {'a': [1, 2], 'b': [1, 2, 3]}
    base_estimator = FastClassifier()
    ratio = 3
    sh = GridSuccessiveHalving(base_estimator, parameters, cv=5,
                               exhaust_budget=True, ratio=ratio,
                               max_budget=max_budget)
    sh.fit(X, y)

    assert sh.n_possible_iterations_ == sh.n_iterations_ == len(sh._r_i_list)
    assert sh._r_i_list == r_i_list
