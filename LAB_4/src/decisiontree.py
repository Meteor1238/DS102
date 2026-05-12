import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf node class

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - np.sum(p ** 2)

    def _best_split(self, X, y):
        n_features = X.shape[1]
        if self.max_features is not None:
            features = np.random.choice(n_features, self.max_features, replace=False)
        else:
            features = np.arange(n_features)

        best_gain, best_feat, best_thresh = -1, None, None
        parent_gini = self._gini(y)

        for f in features:
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                n, nl, nr = len(y), left_mask.sum(), right_mask.sum()
                gain = parent_gini - (nl/n)*self._gini(y[left_mask]) - (nr/n)*self._gini(y[right_mask])
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, f, t

        return best_feat, best_thresh

    def _build(self, X, y, depth):
        if (len(np.unique(y)) == 1 or
                len(y) < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth)):
            return Node(value=np.bincount(y).argmax())

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=np.bincount(y).argmax())

        mask = X[:, feature] <= threshold
        left = self._build(X[mask], y[mask], depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(X, y, 0)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
