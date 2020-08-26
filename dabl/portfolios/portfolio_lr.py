from sklearn.linear_model import LogisticRegression


def portfolio_lr():
    lr = [
        LogisticRegression(C=1, max_iter=10000),
        LogisticRegression(C=0.1, max_iter=10000),
        LogisticRegression(C=10, max_iter=10000),
        LogisticRegression(C=0.01, max_iter=10000),
        LogisticRegression(C=10, max_iter=10000),
        LogisticRegression(C=224.662208931, max_iter=785, solver='liblinear'),
        LogisticRegression(C=0.001, max_iter=10000),
        LogisticRegression(C=0.165918932915, max_iter=57, solver='liblinear'),
        LogisticRegression(C=100000.0, max_iter=724, solver='liblinear'),
        LogisticRegression(C=15.0, solver='liblinear'),
        LogisticRegression(C=0.01, max_iter=10000),
        LogisticRegression(solver='liblinear'),
        LogisticRegression(C=15.0, solver='liblinear'),
        LogisticRegression(C=0.01, solver='liblinear'),
        LogisticRegression(C=500),
        LogisticRegression(C=224.662208931, max_iter=785, solver='liblinear')
    ]
    return (lr)
