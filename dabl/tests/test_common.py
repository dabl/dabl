from sklearn.utils.estimator_checks import check_estimator
from dabl.preprocessing import EasyPreprocessor
import pytest


@pytest.mark.skip(reason="haven't implemented numpy array type checks yet")
def test_preprocessor():
    return check_estimator(EasyPreprocessor)
