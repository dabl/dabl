from dabl.models import FriendlyClassifier
from sklearn.datasets import load_iris


def test_basic():
    iris = load_iris()
    fc = FriendlyClassifier()
    fc.fit(iris.data, iris.target)
