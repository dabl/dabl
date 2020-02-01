import pytest
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, make_blobs, load_boston, load_digits
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from dabl.datasets import load_titanic
from dabl.models import SimpleClassifier, SimpleRegressor, AnyClassifier
from dabl.utils import data_df_from_bunch

iris = load_iris()
X_blobs, y_blobs = make_blobs(centers=2, random_state=0)


def mock_get_estimators_logreg(self):
    return [LogisticRegression(C=0.001), LogisticRegression(C=0.01)]


def mock_get_estimators_dummy(self):
    return [DummyClassifier(strategy='prior')]


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

    # test nupmy array
    er = SimpleRegressor()
    er.fit(boston.data, boston.target)


@pytest.mark.parametrize('Model', [AnyClassifier, SimpleClassifier])
def test_classifier_digits(monkeypatch, Model):
    monkeypatch.setattr(AnyClassifier, '_get_estimators',
                        mock_get_estimators_logreg)
    # regression test for doing clean in fit
    # which means predict can't work
    digits = load_digits()
    X, y = digits.data[::10], digits.target[::10]
    clf = Model().fit(X, y)
    assert clf.score(X, y) > .8


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
    if hasattr(sc, 'predict_proba'):
        assert sc.predict_proba(X_blobs).shape == (X_blobs.shape[0], 2)
    if hasattr(sc, 'decision_function'):
        assert sc.decision_function(X_blobs).shape == (X_blobs.shape[0],)


def get_columns_by_name(ct, name):
    selected_cols = []
    for n, t, cols in ct.transformers_:
        if n == name:
            selected_cols = cols
    if getattr(selected_cols, 'dtype', None) == 'bool':
        selected_cols = selected_cols.index[selected_cols]
    return selected_cols


@pytest.mark.parametrize(
    "type_hints",
    [{'a': 'continuous', 'b': 'categorical', 'c': 'useless'},
     {'a': 'useless', 'b': 'continuous', 'c': 'categorical'},
     ])
@pytest.mark.parametrize('Model', [AnyClassifier, SimpleClassifier])
def test_model_type_hints(monkeypatch, type_hints, Model):
    type_hints = {'a': 'continuous', 'b': 'categorical', 'c': 'useless'}
    X = pd.DataFrame({'a': [0, 1, 0, 1, 0, 0, 1, 0, 1, 1] * 2,
                      'b': [0.1, 0.2, 0.3, 0.1, 0.1, 0.2,
                            0.2, 0.1, 0.1, 0.3] * 2,
                      'c': ['a', 'b', 'a', 'b', 'a', 'a',
                            'b', 'b', 'a', 'b'] * 2,
                      'target': [1, 2, 2, 1, 1, 2, 1, 2, 1, 2] * 2})

    monkeypatch.setattr(Model, '_get_estimators',
                        mock_get_estimators_dummy)

    sc = Model(type_hints=type_hints)
    sc.fit(X, target_col='target')

    ct = sc.est_[0].ct_

    for col_name in type_hints:
        if type_hints[col_name] == 'useless':
            # if a column was annotated as useless it's not used
            assert len(get_columns_by_name(ct, type_hints[col_name])) == 0
        else:
            # if a column was annotated as continuous,
            # it's in the continuous part
            # same for categorical
            assert col_name in get_columns_by_name(ct, type_hints[col_name])


@pytest.mark.parametrize('X, y',
                         [(pd.DataFrame(np.random.random(10).reshape(-1, 1)),
                           pd.Series(np.random.random(10))),
                          (pd.DataFrame(np.random.random(10).reshape(-1, 1)),
                           pd.DataFrame(np.random.random(10).reshape(-1, 1)))])
def test_evaluate_score_ndim(X, y):
    """Test fit() works for both y.ndim == 1 and y.ndim == 2. Two test cases
    are listed in @pytest.mark.parametrize()
    """
    sr = SimpleRegressor(random_state=0)
    print(f"Data ndim: X: {X.shape}, y: {y.shape}")
    sr.fit(X, y)


def test_shuffle_cross_validation():
    # somewhat nonlinear design with sorted target
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 10))
    w = rng.normal(size=(10,))
    y = np.dot(X, w)
    y = .1 * y ** 2 + 2 * y
    # throws off linear model if we sort
    sorting = np.argsort(y)
    X = pd.DataFrame(X[sorting, :])
    y = pd.Series(y[sorting])
    sr = SimpleRegressor(shuffle=False).fit(X, y)
    assert sr.log_[-2].r2 < 0.1
    sr = SimpleRegressor().fit(X, y)
    assert sr.log_[-2].r2 > .9


def test_classification_of_string_targets():
    X = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
    y = np.array(['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b'])
    obj = SimpleClassifier()

    fitted = obj.fit(X, y)
    pred = fitted.predict(np.array([1, 2]).reshape(-1, 1))

    np.testing.assert_array_equal(obj.classes_, np.array(['a', 'b']))
    np.testing.assert_array_equal(pred, np.array(['a', 'b']))
