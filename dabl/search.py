from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.utils import deprecated
from sklearn.model_selection import HalvingGridSearchCV as HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV as \
    HalvingRandomSearchCV


__all__ = ['GridSuccessiveHalving', 'RandomSuccessiveHalving']


@deprecated("GridSuccessiveHalving was upstreamed to sklearn,"
            " please import from sklearn.model_selection.")
class GridSuccessiveHalving(HalvingGridSearchCV):
    pass


@deprecated("RandomSuccessiveHalving was upstreamed to sklearn,"
            " please import from sklearn.model_selection.")
class RandomSuccessiveHalving(HalvingRandomSearchCV):
    pass
