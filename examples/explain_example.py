"""
Model Explanation
=================
"""
from dabl.models import SimpleClassifier

wine_df = dabl.utils.data_df_from_bunch(load_wine())

sc = SimpleClassifier()
sc.fit(wine_df, target_col='target')

dabl.explain(sc)