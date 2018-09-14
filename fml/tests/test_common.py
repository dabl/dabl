from sklearn.utils.estimator_checks import check_estimator
from skltemplate import (TemplateEstimator, TemplateClassifier,
                         TemplateTransformer)


def test_estimator():
    return check_estimator(TemplateEstimator)


def test_classifier():
    return check_estimator(TemplateClassifier)


def test_transformer():
    return check_estimator(TemplateTransformer)
