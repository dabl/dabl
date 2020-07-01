import seaborn as sns
import pandas as pd

from sklearn.datasets import fetch_openml
from dabl.utils import data_df_from_bunch
from dabl.plot.utils import add_counts_to_yticklabels


def test_data_df_from_bunch():
    data_bunch = fetch_openml('MiceProtein')
    data = data_df_from_bunch(data_bunch)
    assert len(data) == len(data_bunch.data)
    assert all(data.target.unique() ==
               ['c-CS-m', 'c-SC-m', 'c-CS-s', 'c-SC-s',
                't-CS-m', 't-SC-m', 't-CS-s', 't-SC-s'])


def test_add_counts_to_yticklabels():
    """Make dummy price df, count housetype frequency, check yticklabels"""
    df = pd.DataFrame({'housetype': ['appt', 'appt', 'house'],
                       'price': [20, 25, 30]})
    target_col = 'price'
    col = 'housetype'
    vc = df[col].value_counts()
    ax = sns.boxplot(x=df[target_col], y=df[col])
    add_counts_to_yticklabels(ax, vc)
    assert ax.get_yticklabels()[0].get_text() == 'appt (2)'
    assert ax.get_yticklabels()[1].get_text() == 'house (1)'
