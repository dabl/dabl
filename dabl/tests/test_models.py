from dabl.models import EasyClassifier
from sklearn.datasets import load_iris


def test_basic():
    iris = load_iris()
    fc = EasyClassifier()
    fc.fit(iris.data, iris.target)
