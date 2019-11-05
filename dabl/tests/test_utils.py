from sklearn.datasets import fetch_openml
from dabl.utils import data_df_from_bunch


def test_data_df_from_bunch():
    data_bunch = fetch_openml('MiceProtein')
    data = data_df_from_bunch(data_bunch)
    assert len(data) == len(data_bunch.data)
    assert all(data.target.unique() ==
               ['c-CS-m', 'c-SC-m', 'c-CS-s', 'c-SC-s',
                't-CS-m', 't-SC-m', 't-CS-s', 't-SC-s'])
