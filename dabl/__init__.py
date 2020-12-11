from .preprocessing import EasyPreprocessor, clean, detect_types
from .models import SimpleClassifier, SimpleRegressor, AnyClassifier
from .plot.supervised import plot
from .explain import explain
from . import datasets

__version__ = "0.2.0-dev"

__all__ = ['EasyPreprocessor', 'SimpleClassifier', 'AnyClassifier',
           'SimpleRegressor',
           'explain', 'clean', 'detect_types', 'plot', 'datasets']
