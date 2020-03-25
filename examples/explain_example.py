"""
Model Explanation
=================
"""
from dabl.models import SimpleClassifier
from dabl.explain import explain
from dabl.datasets import load_wine

wine_df = load_wine()

sc = SimpleClassifier()
sc.fit(wine_df, target_col='target')

explain(sc)


