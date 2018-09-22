from sklearn.utils.estimator_checks import check_estimator
from fml.preprocessing import SimplePreprocessor


def test_preprocessor():
    return check_estimator(SimplePreprocessor)
