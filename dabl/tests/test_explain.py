from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from dabl.datasets import load_titanic
from dabl import SimpleClassifier, EasyPreprocessor, explain, clean


def test_explain_smoke_titanic():
    titanic = load_titanic()
    sc = SimpleClassifier().fit(titanic, target_col='survived')
    explain(sc)
    titanic_clean = clean(titanic)
    sc = SimpleClassifier().fit(titanic_clean, target_col='survived')
    explain(sc)
    X, y = titanic_clean.drop("survived", axis=1), titanic_clean.survived
    ep = EasyPreprocessor()
    preprocessed = ep.fit_transform(X)
    tree = DecisionTreeClassifier().fit(preprocessed, y)
    explain(tree, feature_names=ep.get_feature_names())
    pipe = make_pipeline(EasyPreprocessor(), LogisticRegression())
    pipe.fit(X, y)
    explain(pipe, pipe.steps[0][1].get_feature_names())
