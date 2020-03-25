"""
Model Explanation
=================
"""
from dabl.models import SimpleClassifier
from dabl.explain import explain
from dabl.utils import data_df_from_bunch
from sklearn.datasets import load_wine

wine = load_wine()
wine_df = data_df_from_bunch(wine)

sc = SimpleClassifier()
sc.fit(wine_df, target_col='target')

explain(sc)
