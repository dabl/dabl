import pytest

from sklearn.datasets import make_classification
from sklearn.svm import SVC

from fml.search import GridSuccessiveHalving, RandomSuccessiveHalving


parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
base_estimator = SVC(gamma='scale')


@pytest.mark.parametrize('sh', (
    GridSuccessiveHalving(base_estimator, parameters),
    RandomSuccessiveHalving(base_estimator, parameters, n_iter=4)
))
def test_basic(sh):
    X, y = make_classification(n_samples=1000, random_state=0)
    sh.set_params(random_state=0, cv=5)
    sh.fit(X, y)
    assert sh.score(X, y) > .98
