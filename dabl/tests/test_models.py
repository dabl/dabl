import pytest

from sklearn.datasets import load_iris, make_blobs, load_boston, load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from dabl.datasets import load_titanic
from dabl.models import SimpleClassifier, SimpleRegressor, AnyClassifier
from dabl.utils import data_df_from_bunch

iris = load_iris()
X_blobs, y_blobs = make_blobs(centers=2, random_state=0)


def mock_get_estimators_logreg(self):
    return [LogisticRegression(C=0.001), LogisticRegression(C=0.01)]


@pytest.mark.parametrize("X, y, refit",
                         [(iris.data, iris.target, False),
                          (iris.data, iris.target, True),
                          (X_blobs, y_blobs, False),
                          (X_blobs, y_blobs, False),
                          ])
def test_basic(X, y, refit):
    # test on iris
    ec = SimpleClassifier(refit=refit)
    ec.fit(X, y)
    if refit:
        # smoke test
        ec.predict(X)
    else:
        with pytest.raises(ValueError, match="refit"):
            ec.predict(X)


def test_simple_classifier_titanic():
    titanic = load_titanic()
    ec = SimpleClassifier()
    ec.fit(titanic, target_col='survived')
    ec.predict(titanic.drop('survived', axis=1))


def test_any_classifier_titanic(monkeypatch):
    monkeypatch.setattr(AnyClassifier, '_get_estimators',
                        mock_get_estimators_logreg)
    titanic = load_titanic()
    ac = AnyClassifier()
    ac.fit(titanic, target_col='survived')


def test_regression_boston():
    boston = load_boston()
    data = data_df_from_bunch(boston)
    er = SimpleRegressor()
    er.fit(data, target_col='target')


def test_simplie_classifier_digits():
    # regression test for doing clean in fit
    # which means predict can't work
    digits = load_digits()
    X, y = digits.data[::10], digits.target[::10]
    sc = SimpleClassifier().fit(X, y)
    assert sc.score(X, y) > .8


@pytest.mark.parametrize('model', [LinearSVC(), LogisticRegression()])
def test_delegation_simple(monkeypatch, model):

    def mock_get_estimators(self):
        return [model]
    monkeypatch.setattr(SimpleClassifier, '_get_estimators',
                        mock_get_estimators)
    sc = SimpleClassifier(random_state=0)
    sc.fit(X_blobs, y_blobs)
    assert isinstance(sc.est_[1], type(model))
    assert (hasattr(sc, 'decision_function')
            == hasattr(model, 'decision_function'))
    assert hasattr(sc, 'predict_proba') == hasattr(model, 'predict_proba')
