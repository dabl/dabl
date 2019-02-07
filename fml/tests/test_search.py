from fml.search import SuccessiveHalving
from sklearn.datasets import make_classification
from sklearn.svm import SVC


def test_basic():
    X, y = make_classification(n_samples=1000, random_state=0)
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
    svc = SVC(gamma="scale")
    sh = SuccessiveHalving(svc, parameters, cv=5)

    sh.fit(X, y)
    assert sh.score(X, y) > .98
