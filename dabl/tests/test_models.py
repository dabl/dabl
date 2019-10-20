import pytest
import os
import pandas as pd
from sklearn.datasets import load_iris, make_blobs, load_boston
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from dabl.models import SimpleClassifier, SimpleRegressor
from dabl.utils import data_df_from_bunch

iris = load_iris()
X_blobs, y_blobs = make_blobs(centers=2, random_state=0)


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


def test_dataframe():
    path = os.path.dirname(__file__)
    titanic = pd.read_csv(os.path.join(path, '../datasets/titanic.csv'))[::10]
    ec = SimpleClassifier()
    ec.fit(titanic, target_col='survived')


def test_regression_boston():
    boston = load_boston()
    data = data_df_from_bunch(boston)
    er = SimpleRegressor()
    er.fit(data, target_col='target')


@pytest.mark.parametrize('model', [LinearSVC(), LogisticRegression()])
def test_deletation_simple(monkeypatch, model):
    def mock_get_estimators_linearsvc(self):
        return [model]

    monkeypatch.setattr(SimpleClassifier, '_get_estimators',
                        mock_get_estimators_linearsvc)
    sc = SimpleClassifier(random_state=0)
    sc.fit(X_blobs, y_blobs)
    assert isinstance(sc.est_[1], type(model))
    assert (hasattr(sc, 'decision_function')
            == hasattr(model, 'decision_function'))
    assert hasattr(sc, 'predict_proba') == hasattr(model, 'predict_proba')
