
import itertools
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid
from scipy.sparse import csgraph

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import log_loss, confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import (
    cross_val_score, StratifiedShuffleSplit, cross_val_predict)
from sklearn.cluster import SpectralClustering


import scipy
from sklearn.utils.fixes import logsumexp
from sklearn.preprocessing import LabelBinarizer, LabelEncoder


def class_entropy(X, y, col1, col2, n_bins=40):
    groups = X.groupby(y)
    all_counts = []
    x1 = X[col1]
    x2 = X[col2]
    for name, group in groups:
        counts, binx, biny = np.histogram2d(group[col1], group[col2],
                                            range=[[x1.min(), x1.max()],
                                                   [x2.min(), x2.max()]],
                                            bins=n_bins)
        all_counts.append(counts)
    all_counts = np.array(all_counts)
    all_counts = all_counts.reshape(all_counts.shape[0], -1)
    all_counts = np.maximum(all_counts, .1)
    all_counts = all_counts / all_counts.sum(axis=0)
    entropy = -np.mean(np.log(all_counts) * all_counts)
    return entropy


def _find_scatter_plots_classification_entropy(X, target, how_many=3,
                                               n_bins=40):
    scores = []
    for col1, col2 in itertools.combinations(X.columns, 2):
        res = -class_entropy(X, target, col1, col2, n_bins=n_bins)
        scores.append((col1, col2, res))

    scores = pd.DataFrame(scores, columns=['feature0', 'feature1', 'score'])
    top = scores.sort_values(by='score').iloc[-how_many:][::-1]
    return top


def _find_scatter_plots_classification_gb(X, y, max_depth=3,
                                          max_leaf_nodes=None,
                                          learning_rate=1,
                                          how_many=3, verbose=0):
    y = LabelEncoder().fit_transform(y)
    gb = MGradientBoostingClassifierPairs(
        n_iter=how_many, learning_rate=learning_rate, max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes, verbose=verbose)
    gb.fit(X, y)

    res = pd.DataFrame(gb.feature_pairs, columns=['feature0', 'feature1'])
    res['score'] = -np.array(gb.scores)
    return res


class BinaryCrossEntropy:
    def initial_scores(self, y_true):
        ret = scipy.special.logit(y_true.mean())
        return np.repeat(ret, y_true.shape[0])

    def compute_gradients(self, y_true, y_pred):
        return sigmoid(y_pred) - y_true

    def raw_predictions_to_proba(self, raw_predictions):
        return sigmoid(raw_predictions)


class MultinomialCrossEntropy:
    def initial_scores(self, y):
        ret = np.log(np.bincount(y) / y.sum())
        return np.repeat(ret.reshape(1, -1), y.shape[0], axis=0)

    def compute_gradients(self, y_true, y_scores_pred):
        y_scores_true = LabelBinarizer().fit_transform(y_true)

        return (np.exp(y_scores_pred
                       - logsumexp(y_scores_pred, axis=1).reshape(-1, 1))
                - y_scores_true)

    def raw_predictions_to_proba(self, raw_predictions):
        return scipy.special.softmax(raw_predictions, axis=1)


class BaseGradientBoostingPairs(BaseEstimator):
    def __init__(self, n_iter, learning_rate, loss, max_depth=3,
                 max_leaf_nodes=None, verbose=0):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.verbose = verbose

    def fit(self, X, y):
        y_pred_train = self.loss.initial_scores(y)

        self.predictors = list()
        self.feature_pairs = []
        self.scores = []
        for m in range(self.n_iter):  # Gradient Descent
            negative_gradient = -self.loss.compute_gradients(y, y_pred_train)
            these_predictors = []
            these_scores = []
            these_pairs = []
            for i, j in itertools.combinations(range(X.shape[1]), 2):
                if (i, j) in self.feature_pairs:
                    continue
                this_X = X[:, [i, j]]
                new_predictors = []
                predictions = []
                for k in range(negative_gradient.shape[1]):
                    new_predictor = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        max_leaf_nodes=self.max_leaf_nodes)
                    new_predictor.fit(
                        this_X, y=self.learning_rate * negative_gradient[:, k])
                    new_predictors.append(new_predictor)
                    predictions.append(new_predictor.predict(this_X))
                these_predictors.append(new_predictors)
                predictions = np.c_[predictions].T
                this_proba = self.loss.raw_predictions_to_proba(
                    y_pred_train + predictions)
                these_scores.append(log_loss(y, this_proba))
                these_pairs.append((i, j))
            best_idx = np.argmin(these_scores)
            if self.verbose:
                print(these_scores[best_idx])
            self.scores.append(these_scores[best_idx])
            best_predictors = these_predictors[best_idx]
            self.feature_pairs.append(these_pairs[best_idx])
            this_X = X[:, these_pairs[best_idx]]
            predictions = []
            for predictor in best_predictors:
                predictions.append(predictor.predict(this_X))
            predictions = np.c_[predictions].T

            y_pred_train += predictions

            self.predictors.append(best_predictors)
            if np.mean(y == self.predict(X)) > 0.99:
                # stop once we overfitted
                break
            if self.verbose:
                print(confusion_matrix(y, self.predict(X)))

    def predict(self, X):
        predictions = np.zeros((len(X), len(self.predictors[0])))
        for i, (predictors, pair) in enumerate(zip(self.predictors,
                                                   self.feature_pairs)):
            for j, predictor in enumerate(predictors):
                predictions[:, j] += predictor.predict(X[:, pair])
        return predictions


class MGradientBoostingClassifierPairs(BaseGradientBoostingPairs,
                                       ClassifierMixin):

    def __init__(self, n_iter=100, learning_rate=.1, max_depth=3,
                 max_leaf_nodes=None, verbose=0):
        super().__init__(n_iter, learning_rate, loss=MultinomialCrossEntropy(),
                         max_depth=max_depth, max_leaf_nodes=max_leaf_nodes,
                         verbose=verbose)

    def predict(self, X):
        raw_predictions = super().predict(X)
        proba_positive_class = self.loss.raw_predictions_to_proba(
            raw_predictions)
        return np.argmax(proba_positive_class, axis=1)

    def decision_function(self, X):
        return super().predict(X)


def _find_scatter_plots_classification(X, target, how_many=3):
    # input is continuous
    # look at all pairs of features, find most promising ones
    # dummy = DummyClassifier(strategy='prior').fit(X, target)
    # baseline_score = recall_score(target, dummy.predict(X), average='macro')
    scores = []
    # converting to int here might save some time
    _, target = np.unique(target, return_inverse=True)
    # limit to 2000 training points for speed?
    train_size = min(2000, int(.9 * X.shape[0]))
    cv = StratifiedShuffleSplit(n_splits=3, train_size=train_size)
    for i, j in itertools.combinations(np.arange(X.shape[1]), 2):
        this_X = X[:, [i, j]]
        # assume this tree is simple enough so not be able to overfit in 2d
        # so we don't bother with train/test split
        tree = DecisionTreeClassifier(max_leaf_nodes=8)
        scores.append((i, j, np.mean(cross_val_score(
            tree, this_X, target, cv=cv, scoring='recall_macro'))))

    scores = pd.DataFrame(scores, columns=['feature0', 'feature1', 'score'])
    top = scores.sort_values(by='score').iloc[-how_many:][::-1]
    return top


def decompose_confusion_matrix(cm):
    # test this soo much!
    n, connected_components = csgraph.connected_components(cm)
    components_sizes = np.bincount(connected_components)
    if np.sum(components_sizes >= 2) >= 2:
        # we have at least two components of size at least two
        return connected_components
    if components_sizes.max() <= 2:
        # can't really split any more
        return connected_components
    # split largest connected component in two
    largest_component = np.argmax(components_sizes)
    component_mask = connected_components == largest_component
    cm_sub = cm[:, component_mask][component_mask, :]
    sc = SpectralClustering(n_clusters=2, affinity='precomputed')
    relabels = sc.fit_predict(cm_sub + cm_sub.T)
    connected_components[component_mask] = (
        relabels + np.max(connected_components) + 1)
    return np.unique(connected_components, return_inverse=True)[1]


def hierarchical_cm(X, y, verbose=0):
    classes = np.unique(y)
    n_classes = len(classes)
    if n_classes < 2:
        print("n_classes < 2")
        return []
    res = _find_scatter_plots_classification(X, y, how_many=1)
    feature0, feature1 = res.iloc[0, :-1].astype('int')
    X_this = X[:, [feature0, feature1]]

    preds = cross_val_predict(DecisionTreeClassifier(max_depth=5), X_this, y)
    cm = confusion_matrix(y, preds, normalize='true')
    if verbose > 0:
        print(cm)
    children = [(classes, feature0, feature1, cm)]
    if n_classes < 3:
        # only have two classes, no need to split any more
        return children
    labels = decompose_confusion_matrix(cm)
    cluster_sizes = np.bincount(labels)
    for i, s in enumerate(cluster_sizes):
        if s < 2:
            continue
        cluster_classes = classes[labels == i]
        mask = y.isin(cluster_classes)
        X_new = X[mask]
        y_new = y[mask]
        res = hierarchical_cm(X_new, y_new)
        children.extend(res)
    return children
