import numpy as np
from sklearn.metrics import adjusted_rand_score
from dabl.plot.quality_measures import decompose_confusion_matrix


def test_decompose_confusion_matrix():
    # connected component conditions
    # no component has size more than 2
    cm = np.eye(5)
    assert np.all(decompose_confusion_matrix(cm) == np.arange(5))
    cm[0, 1] = 1
    assert np.all(decompose_confusion_matrix(cm) == [0, 0, 1, 2, 3])
    cm[2, 3] = 1
    np.all(decompose_confusion_matrix(cm) == [0, 0, 1, 1, 2])
    # we have two components that are "large enough"
    cm[3, 4] = 1
    np.all(decompose_confusion_matrix(cm) == [0, 0, 1, 1, 1])
    # singleton component and second components of size 4:
    # need spectral clustering now
    cm[3, 4] = 0
    cm[1, 2] = 1
    res = decompose_confusion_matrix(cm)
    assert adjusted_rand_score(res, [1, 1, 2, 2, 0]) == 1
