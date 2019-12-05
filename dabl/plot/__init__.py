from .supervised import (plot, plot_classification_categorical,
                         plot_regression_categorical,
                         plot_classification_continuous,
                         plot_regression_continuous,
                         class_hists)
from .utils import (find_pretty_grid, mosaic_plot, discrete_scatter,
                    plot_coefficients)


__all__ = [
    'find_pretty_grid', 'plot', 'plot_classification_categorical',
    'plot_classification_continuous', 'plot_regression_categorical',
    'plot_regression_continuous', 'mosaic_plot', 'discrete_scatter',
    'plot_coefficients', 'class_hists']
