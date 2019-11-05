from .preprocessing import EasyPreprocessor, clean, detect_types
from .models import SimpleClassifier, SimpleRegressor
from .plot.supervised import plot
from .explain import explain
from . import datasets

__version__ = "0.1.1"

__all__ = ['EasyPreprocessor', 'SimpleClassifier', 'SimpleRegressor',
           'explain', 'clean', 'detect_types', 'plot', 'datasets']
