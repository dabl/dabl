import pytest
import os
import pandas as pd
from sklearn.datasets import load_iris, make_blobs

from dabl.models import SimpleClassifier

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
