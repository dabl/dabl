from sklearn.utils.estimator_checks import check_estimator
from fml.preprocessing import SimplePreprocessor
import pytest


@pytest.mark.skip(reason="haven't implemented numpy array type checks yet")
def test_preprocessor():
    return check_estimator(SimplePreprocessor)
