from numpy.random import RandomState
import numpy as np


class Node:
    def __init__(
        self,
        feature: int,
        split_value: float,
        left: "Node | None" = None,
        right: "Node | None" = None,
        size: int = 0,
    ):
        """
        Initializes a Node in the Isolation Tree.
        Args:
            feature (int): The feature index used for splitting.
            split_value (float): The value used for splitting the feature.
            left (Node): The left child node.
            right (Node): The right child node.
            size (int): The number of samples in the node.
        """
        self.feature: int = feature
        self.split_value: float = split_value
        self.left: Node | None = left
        self.right: Node | None = right
        self.size: int = size

    def __repr__(self):
        return f"Node(feature={self.feature}, split_value={self.split_value}, size={self.size})"

    def is_leaf(self):
        return self.left is None and self.right is None

    def partition(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Partitions the data X based on the node's feature and split value.

        Args:
            X (array-like): The input samples.
        Returns:
            left_partition (array): Samples that go to the left child.
            right_partition (array): Samples that go to the right child.
        """
        left_partition = X[X[:, self.feature] < self.split_value]
        right_partition = X[X[:, self.feature] >= self.split_value]
        return left_partition, right_partition


class IsolationTree:
    def __init__(
        self,
        max_samples: int,
        max_features: int,
        random_state: int | RandomState | None = None,
        max_height: int = 10,
    ):
        """
        Initializes the IsolationTree.

        Args:
            max_samples (int): The number of samples to draw from X to train the tree.
            max_features (int): The number of features to draw from X to train the tree.
            random_state (int or RandomState instance): Controls the randomness of the estimator.
        """
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_height = max_height
        self.random_state = random_state
        self._set_random_state()

    def _set_random_state(self):
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif isinstance(self.random_state, RandomState):
            np.random.set_state(self.random_state.get_state())

    def _subsample(self, X) -> np.ndarray:
        """
        Subsamples the input data X.

        Args:
            X (array-like): The input samples.

        Returns:
            subsample (array): The subsampled data.
        """
        if X.shape[0] <= self.max_samples:
            return X
        idx = np.random.choice(X.shape[0], size=self.max_samples, replace=False)
        return X[idx]

    def _fit_node(self, X, current_height: int, max_height: int) -> Node:
        """
        Recursively fits a node in the IsolationTree.

        Args:
            X (array-like): The input samples.
            current_height (int): The current height of the node.
            max_height (int): The maximum height of the tree.

        """
        if (current_height >= max_height) or (X.shape[0] <= 1) or np.all(X == X[0]):
            return Node(feature=-1, split_value=-1, size=X.shape[0])
        # random feature selection
        feature = np.random.choice(self.max_features)
        # random split value selection
        split_value = np.random.uniform(np.min(X[:, feature]), np.max(X[:, feature]))
        node = Node(feature=feature, split_value=split_value, size=X.shape[0])
        # data partitioning
        left_partition, right_partition = node.partition(X)
        # recursion
        node.left = self._fit_node(left_partition, current_height + 1, max_height)
        node.right = self._fit_node(right_partition, current_height + 1, max_height)
        return node

    def fit(self, X, sample_weight=None) -> None:
        """
        Fits the IsolationTree to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        # create root node w/ subsample
        self.sample = self._subsample(X)
        self.root_ = self._fit_node(
            self.sample, current_height=0, max_height=self.max_height
        )
        return None

    def trace_path(
        self, x: np.ndarray, node: Node | None = None, current_height: int = 0
    ) -> int:
        """
        Traces the path length of a single sample x through the tree.

        Args:
            x (array-like): The input sample.
            node (Node, optional): The current node in the tree.
            current_height (int): The current height in the tree.

        Returns:
            path_length (int): The path length for the sample x.
        """
        if node is None:
            if self.root_ is None:
                raise ValueError("The tree has not been fitted yet.")
            node = self.root_
        if node.is_leaf():
            return current_height
        if x[node.feature] < node.split_value:
            return self.trace_path(x, node.left, current_height + 1)
        else:
            return self.trace_path(x, node.right, current_height + 1)

    def path_length(self, X) -> np.ndarray:
        """
        Computes the path length for each sample in X.

        Args:
            X (array-like): The input samples.

        Returns:
            path_lengths (array): The path lengths for each sample.
        """
        return np.array([self.trace_path(x) for x in X])


class IsolationForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int | str = "auto",
        contamination: float | str = "auto",
        max_features: int = 1,
        bootstrap: bool = False,
        random_state: int | RandomState | None = None,
        verbose: bool = False,
        warm_start: bool = False,
    ):
        """
        Initializes the IsolationForest model.

        Args:
            n_estimators (int): The number of base estimators (isolation trees) in the ensemble.
            max_samples (int or 'auto'): The number of samples to draw from X to train each base estimator.
            contamination (float or 'auto'): The amount of contamination (proportion of outliers in the data set).
            max_features (int): The number of features to draw from X to train each base estimator.
            bootstrap (bool): Whether bootstrap samples are used when building trees.
            random_state (int or RandomState instance): Controls the randomness of the estimator.
            verbose (bool): Enable verbose output.
            warm_start (bool): When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.
        """
        # Initialize attributes based on parameters
        self.n_estimators: int = n_estimators
        self.max_samples: int = (
            max_samples
            if isinstance(max_samples, int)
            else self._determine_max_samples()
        )
        self.contamination: float = (
            contamination if isinstance(contamination, float) else 0.1
        )
        self.max_features: int = max_features
        self.bootstrap: bool = bootstrap
        self.random_state: int | RandomState = (
            random_state if random_state is not None else RandomState()
        )
        self.verbose: bool = verbose
        self.warm_start: bool = warm_start
        self.estimators_: list[IsolationTree] = []

    def _determine_max_samples(self) -> int:
        """
        Determines the maximum number of samples to draw from X
        based on the dataset size.

        Returns:
            max_samples (int): The determined maximum number of samples.
        """
        return 256

    def fit(self, X, sample_weight=None):
        """
        Fits the IsolationForest model to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        if not self.warm_start or not self.estimators_:
            self.estimators_ = [
                self._make_estimator() for _ in range(self.n_estimators)
            ]
            for estimator in self.estimators_:
                estimator.fit(X, sample_weight=sample_weight)
        else:
            new_estimators = [self._make_estimator() for _ in range(self.n_estimators)]
            for estimator in new_estimators:
                estimator.fit(X, sample_weight=sample_weight)
            self.estimators_.extend(new_estimators)

    def _anomaly_score(self, x: np.ndarray) -> float:
        """
        Computes the anomaly score for a single sample x.

        Args:
            x (array-like): The input sample.
        Returns:
            score (float): The anomaly score for the sample x.
        """
        return float(
            np.mean([estimator.trace_path(x) for estimator in self.estimators_])
        )

    def decision_function(self, X) -> np.ndarray:
        """
        Average anomaly score of X of the base classifiers.

        Parameters:
            X (array-like): The input samples.

        Returns:
            scores (array): The anomaly scores for each sample.
        """
        return np.array([self._anomaly_score(x) for x in X])

    def predict(self, X) -> np.ndarray:
        """
        Predicts if a particular sample is an outlier or not.

        Args:
            X (array-like): The input samples.

        Returns:
            predictions (array): Array of -1 for outliers and 1 for inliers.
        """
        scores = self.decision_function(X)
        threshold = np.percentile(scores, 100 * self.contamination)
        return np.where(scores >= threshold, -1, 1)

    def fit_predict(self, X, sample_weight=None) -> np.ndarray:
        """
        Fits the model to the training data and returns the predictions.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        Returns:
            predictions (array): Array of -1 for outliers and 1 for inliers.
        """
        self.fit(X, sample_weight=sample_weight)
        return self.predict(X)

    def _make_estimator(self) -> IsolationTree:
        """
        Helper method to create a single Isolation Tree (base estimator).
        """
        return IsolationTree(
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state,
        )
