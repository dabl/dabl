import os
import string
import random
import pytest

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from dabl.preprocessing import (detect_types, EasyPreprocessor,
                                DirtyFloatCleaner, clean)


X_cat = pd.DataFrame({'a': ['b', 'c', 'b'],
                      'second': ['word', 'no', ''],
                      'binary': [0.0, 1, 0]})

# FIXME two float values is not a float but binary!
# FIXME features that are always missing are constant!
# FIXME need to test dealing with categorical dtype
# FIXME make sure in plotting single axes objects work everywhere (ravel issue)
# FIXME Fail early on duplicate column names!!!
# TODO add test that weird missing values in strings are correctly interpreted
# TODO test non-trivial case of EasyPreprocessor?!"!"


def make_dirty_float():
    rng = np.random.RandomState(0)
    cont_clean = ["{:2.2f}".format(x) for x in rng.uniform(size=100)]
    dirty = pd.DataFrame(cont_clean, columns=['a_column'])
    dirty[::12] = "missing"
    dirty.iloc[3, 0] = "garbage"
    return dirty


def test_duplicate_columns():
    X = pd.DataFrame([[0, 1]], columns=['a', 'a'])
    with pytest.raises(ValueError, match="Duplicate Columns"):
        clean(X)

    with pytest.raises(ValueError, match="Duplicate Columns"):
        detect_types(X)


def test_duplicate_index():
    X = X_cat.copy()
    X.index = np.ones(len(X), np.int)
    assert not X.index.is_unique
    with pytest.raises(ValueError):
        detect_types(X)
    with pytest.warns(UserWarning):
        X_clean = clean(X)
    assert X_clean.index.is_unique


def test_detect_constant():
    X = pd.DataFrame({'a': [0, 0, 0, 0],
                      'second': ['no', 'no', 'no', 'no'],
                      'b': [0.0, 0.0, 0.0, 0],
                      'weird': ['0', '0', '0', '0']})
    with pytest.warns(UserWarning, match="Discarding near constant"):
        res = detect_types(X)
    assert res.useless.sum() == 4


def test_detect_types():
    def random_str(length=7):
        return "".join([random.choice(string.ascii_letters)
                        for i in range(length)])

    near_constant_float = np.repeat(np.pi, repeats=100)
    near_constant_float[:2] = 0

    df_all = pd.DataFrame(
        {'categorical_string': ['a', 'b'] * 50,
         'binary_int': np.random.randint(0, 2, size=100),
         'categorical_int': np.random.randint(0, 4, size=100),
         'low_card_float': np.random.randint(0, 4, size=100).astype(np.float),
         'binary_float': np.random.randint(0, 2, size=100).astype(np.float),
         'cont_int': np.repeat(np.arange(50), 2),
         'unique_string': [random_str() for i in range(100)],
         'continuous_float': np.random.normal(size=100),
         'constant_nan': np.repeat(np.NaN, repeats=100),
         'constant_string': ['every_day'] * 100,
         'constant_float': np.repeat(np.pi, repeats=100),
         'near_constant_float': near_constant_float,
         'index_0_based': np.arange(100),
         'index_1_based': np.arange(1, 101),
         'index_shuffled': np.random.permutation(100)
         })
    with pytest.warns(UserWarning, match="Discarding near constant"):
        res = detect_types(df_all)
    types = res.T.idxmax()
    assert types['categorical_string'] == 'categorical'
    assert types['binary_int'] == 'categorical'
    assert types['categorical_int'] == 'categorical'
    # assert types['low_card_int_binomial'] == 'continuous'
    # a bit inconsistent: we're treating cardinality 2
    # floats as categorical but cardinality 3 or 4 not
    assert types['low_card_float'] == 'continuous'
    assert types['binary_float'] == 'categorical'
    assert types['cont_int'] == 'continuous'
    assert types['unique_string'] == 'free_string'
    assert types['continuous_float'] == 'continuous'
    assert types['constant_nan'] == 'useless'
    assert types['constant_string'] == 'useless'
    assert types['constant_float'] == 'useless'
    assert types['near_constant_float'] == 'useless'
    assert types['index_0_based'] == 'useless'
    assert types['index_1_based'] == 'useless'
    # Not detecting a shuffled index right now :-/
    assert types['index_shuffled'] == 'continuous'

    res = detect_types(X_cat)
    assert len(res) == 3
    assert res.categorical.all()
    assert ~res.continuous.any()

    iris = load_iris()
    res_iris = detect_types(pd.DataFrame(iris.data))
    assert (res_iris.sum(axis=1) == 1).all()
    assert res_iris.continuous.sum() == 4


def test_detect_low_cardinality_int():
    df_all = pd.DataFrame(
        {
         'binary_int': np.random.randint(0, 2, size=1000),
         'categorical_int': np.random.randint(0, 4, size=1000),
         'low_card_int_uniform': np.random.randint(0, 20, size=1000),
         'low_card_int_binomial': np.random.binomial(20, .3, size=1000),
         'cont_int': np.repeat(np.arange(500), 2),
        })

    res = detect_types(df_all)
    types = res.T.idxmax()

    assert types['binary_int'] == 'categorical'
    assert types['categorical_int'] == 'categorical'
    assert types['low_card_int_uniform'] == 'low_card_int'
    assert types['low_card_int_binomial'] == 'low_card_int'
    assert types['cont_int'] == 'continuous'


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
    # only dirty:
    res = detect_types(pd.DataFrame(dirty))
    assert len(res) == 1
    assert res.dirty_float[0]

    # dirty and clean
    X = pd.DataFrame({'a': cont_clean, 'b': dirty})
    res = detect_types(X)
    assert len(res) == 2
    assert res.continuous['a']
    assert ~res.continuous['b']
    assert ~res.dirty_float['a']
    assert res.dirty_float['b']


def test_transform_dirty_float():
    dirty = make_dirty_float()
    dfc = DirtyFloatCleaner()
    dfc.fit(dirty)
    res = dfc.transform(dirty)
    # TODO test for new values in test etc
    assert res.shape == (100, 3)
    assert (res.dtypes == float).all()
    assert res.a_column_missing.sum() == 9
    assert res.a_column_garbage.sum() == 1


def test_simple_preprocessor():
    sp = EasyPreprocessor()
    sp.fit(X_cat)
    trans = sp.transform(X_cat)
    assert trans.shape == (3, 7)  # FIXME should be 6?

    iris = load_iris()
    sp = EasyPreprocessor()
    sp.fit(iris.data)


def test_simple_preprocessor_dirty_float():
    dirty = pd.DataFrame(make_dirty_float())
    fp = EasyPreprocessor()
    fp.fit(dirty)
    res = fp.transform(dirty)
    assert res.shape == (100, 3)
    rowsum = res.sum(axis=0)
    # count of "garbage"
    assert rowsum[1] == 1
    # count of "missing"
    assert rowsum[2] == 9

    # make sure we can transform a clean column
    fp.transform(pd.DataFrame(['0', '1', '2'], columns=['a_column']))


def test_titanic_detection():
    path = os.path.dirname(__file__)
    titanic = pd.read_csv(os.path.join(path, 'titanic.csv'))
    types_table = detect_types(titanic)
    types = types_table.T.idxmax()
    true_types = [
        'dirty_float', 'categorical', 'dirty_float', 'free_string',
        'categorical', 'dirty_float', 'free_string', 'free_string',
        'low_card_int', 'categorical',
        'categorical', 'low_card_int', 'categorical', 'free_string']
    assert (types == true_types).all()
    titanic_nan = pd.read_csv(os.path.join(path, 'titanic.csv'), na_values='?')
    types_table = detect_types(titanic_nan)
    types = types_table.T.idxmax()
    true_types_clean = [t if t != 'dirty_float' else 'continuous'
                        for t in true_types]
    assert (types == true_types_clean).all()
