import pytest

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_diabetes

from dabl.datasets import load_titanic
from dabl import SimpleClassifier, EasyPreprocessor, explain, clean
from dabl import SimpleRegressor


def test_explain_smoke_titanic():
    titanic = load_titanic()
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
    explain(pipe, feature_names=pipe[0].get_feature_names())


@pytest.mark.parametrize("model", [LogisticRegression(C=0.1),
                                   DecisionTreeClassifier(max_depth=5),
                                   RandomForestClassifier(n_estimators=10)])
def test_explain_titanic_val(model):
    # add multi-class
    # add regression
    titanic = load_titanic()
    titanic_clean = clean(titanic)
    X, y = titanic_clean.drop("survived", axis=1), titanic_clean.survived
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y,
                                                      random_state=42)
    pipe = make_pipeline(EasyPreprocessor(), model)
    pipe.fit(X_train, y_train)
    # without validation set
    explain(pipe, feature_names=X.columns)
    # with validation set
    explain(pipe, X_val, y_val, feature_names=X.columns)


def test_explain_wine_smoke():
    # multi class classification
    wine = load_wine()
    X_train, X_val, y_train, y_val = train_test_split(
        wine.data, wine.target, stratify=wine.target, random_state=42)
    sc = SimpleClassifier().fit(X_train, y_train)
    explain(sc, X_val, y_val)


def test_explain_diabetes_smoke():
    diabetes = load_diabetes()
    X_train, X_val, y_train, y_val = train_test_split(
        diabetes.data, diabetes.target, random_state=42)
    sc = SimpleRegressor().fit(X_train, y_train)
    explain(sc, X_val, y_val)
