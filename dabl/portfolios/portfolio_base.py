from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC

enable_hist_gradient_boosting


def portfolio_base():
    base = [
        LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=10, solver='lbfgs', multi_class='multinomial'),
        LogisticRegression(C=.1, solver='lbfgs', multi_class='multinomial'),
        RandomForestClassifier(max_features=None, n_estimators=100),
        RandomForestClassifier(max_features='sqrt', n_estimators=100),
        RandomForestClassifier(max_features='log2', n_estimators=100),
        SVC(C=1, gamma=0.03, kernel='rbf'),
        SVC(C=1, gamma='scale', kernel='rbf'),
        HistGradientBoostingClassifier()
    ]
    return (base)
