
import itertools
import pandas as pd
import numpy as np
from scipy.special import expit as sigmoid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import log_loss, confusion_matrix

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
                                          how_many=3):
    y = LabelEncoder().fit_transform(y)
    gb = MGradientBoostingClassifierPairs(
        n_iter=how_many, learning_rate=learning_rate, max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes)
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
                 max_leaf_nodes=None):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes

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
                new_predictor = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    max_leaf_nodes=self.max_leaf_nodes)
                new_predictor.fit(this_X, y=self.learning_rate * negative_gradient)
                these_predictors.append(new_predictor)
                this_proba = self.loss.raw_predictions_to_proba(y_pred_train + new_predictor.predict(this_X))
                these_scores.append(log_loss(y, this_proba))
                these_pairs.append((i, j))
            best_idx = np.argmin(these_scores)
            print(these_scores[best_idx])
            self.scores.append(these_scores[best_idx])
            new_predictor = these_predictors[best_idx]
            self.feature_pairs.append(these_pairs[best_idx])
            this_X = X[:, these_pairs[best_idx]]
            y_pred_train += new_predictor.predict(this_X)

            self.predictors.append(new_predictor)  # save for predict()
            print(confusion_matrix(y, self.predict(X)))

    def predict(self, X):
        return sum(predictor.predict(X[:, pair])
                   for predictor, pair in zip(self.predictors,
                                              self.feature_pairs))


class MGradientBoostingClassifierPairs(BaseGradientBoostingPairs,
                                       ClassifierMixin):

    def __init__(self, n_iter=100, learning_rate=.1, max_depth=3,
                 max_leaf_nodes=None):
        super().__init__(n_iter, learning_rate, loss=MultinomialCrossEntropy(),
                         max_depth=max_depth, max_leaf_nodes=max_leaf_nodes)

    def predict(self, X):
        raw_predictions = super().predict(X)
        proba_positive_class = self.loss.raw_predictions_to_proba(
            raw_predictions)
        return np.argmax(proba_positive_class, axis=1)

    def decision_function(self, X):
        return super().predict(X)
