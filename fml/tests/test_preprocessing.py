from fml.preprocessing import (detect_types_dataframe, SimplePreprocessor,
                               DirtyFloatCleaner)
import pandas as pd
import numpy as np

X_cat = pd.DataFrame({'a': ['b', 'c', 'b'],
                      'second': ['word', 'no', '']})


def test_detect_types_dataframe():
    res = detect_types_dataframe(X_cat)
    assert len(res) == 2
    assert res.categorical.all()
    assert ~res.continuous.any()


def test_detect_string_floats():
    # test if we can find floats that are encoded as strings
    # sometimes they have weird missing values
    rng = np.random.RandomState(0)
    cont_clean = ["{:2.2f}".format(x) for x in rng.uniform(size=100)]
    dirty = pd.Series(cont_clean)
    # FIXME hardcoded frequency of tolerated missing
    # FIXME add test with integers
    # FIXME whitespace?
    dirty[::12] = "missing"
    X = pd.DataFrame({'a': cont_clean, 'b': dirty})
    res = detect_types_dataframe(X)
    assert len(res) == 2
    assert res.continuous['a']
    assert ~res.continuous['b']
    assert ~res.dirty_float_string['a']
    assert res.dirty_float_string['b']


def test_transform_dirty_float():
    rng = np.random.RandomState(0)
    cont_clean = ["{:2.2f}".format(x) for x in rng.uniform(size=100)]
    dirty = pd.DataFrame(cont_clean)
    dirty[::12] = "missing"
    dirty.iloc[3, 0] = "garbage"
    dfc = DirtyFloatCleaner()
    dfc.fit(dirty)
    res = dfc.transform(dirty)
    # TODO test for new values in test etc
    assert res.shape == (100, 3)
    assert (res.dtypes == float).all()
    assert res.x0_missing.sum() == 9
    assert res.x0_garbage.sum() == 1


def test_simple_preprocessor():
    sp = SimplePreprocessor()
    sp.fit(X_cat)
    trans = sp.transform(X_cat)
    assert trans.shape == (3, 5)


# TODO add tests that floats as strings are correctly interpreted
# TODO add test that weird missing values in strings are correctly interpreted
# TODO check that we detect ID columns
