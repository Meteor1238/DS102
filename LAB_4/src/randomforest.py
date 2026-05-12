import numpy as np
from decisiontree import DecisionTree

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        max_features = (int(np.sqrt(n_features)) if self.max_features == 'sqrt'
                        else int(np.log2(n_features)) if self.max_features == 'log2'
                        else self.max_features)

        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(preds[:, i].astype(int)).argmax() for i in range(X.shape[0])])
