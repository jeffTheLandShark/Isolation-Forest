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

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def partition(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Partitions the data X based on the node's feature and split value.

        Args:
            X (array-like): The input samples.
        Returns:
            left_partition, right_partition (tuple): The left and right partitions of X.
        Raises:
            ValueError: If the feature index is invalid.
        """
        if X.size == 0:
            return X, X  # empty partitions
        if self.feature < 0 or self.feature >= X.shape[1]:
            # if feature is invalid, don't split
            raise ValueError("Invalid feature index for partitioning")
        left_part = X[:, self.feature] < self.split_value  # bool mask
        return X[left_part], X[~left_part]


class IsolationTree:
    def __init__(
        self, max_samples: int = 256, random_state: Optional[int | RandomState] = None
    ):
        """
        Initializes the IsolationTree.
        Args:
            max_samples (int): The maximum number of samples to use for building the tree.
            random_state (int | RandomState | None): Seed or RandomState
        """
        if max_samples <= 0:
            raise ValueError("max_samples must be > 0")
        self.max_samples: int = max_samples
        self.max_height: int = int(np.ceil(np.log2(self.max_samples)))
        # internal RNG: always a RandomState instance
        if isinstance(random_state, RandomState):
            self._rng = RandomState(random_state.randint(0, 2**31 - 1))
        else:
            self._rng = RandomState(random_state)
        self._root: Node | None = None
        self._num_features: Optional[int] = None
        self.n_samples: Optional[int] = None

    def _fit_node(self, X: np.ndarray, current_height: int) -> Node:
        """
        Recursively fits a node in the IsolationTree.

        Args:
            X (array-like): The input samples.
            current_height (int): The current height of the node.
            max_height (int): The maximum height of the tree.

        """
        if self._num_features is None:
            raise ValueError("_num_features must be set before building nodes")
        n = X.shape[0]
        if (current_height >= self.max_height) or (n <= 1) or np.all(X == X[0]):
            return Node(size=n)
        # random feature selection
        rand_feature = int(self._rng.choice(self._num_features))
        feature_vals = X[:, rand_feature]
        if np.all(np.isclose(feature_vals, feature_vals[0], atol=EPSILON)):
            # all values are (approximately) the same -> no split possible
            return Node(feature=rand_feature, split_value=feature_vals[0], size=n)
        split_val = float(self._rng.uniform(np.min(feature_vals), np.max(feature_vals)))
        node = Node(feature=rand_feature, split_value=split_val, size=n)
        left_X, right_X = node.partition(X)
        # recursion
        node.left = self._fit_node(left_X, current_height + 1)
        node.right = self._fit_node(right_X, current_height + 1)
        return node

    def feature_importances_(self) -> np.ndarray:
        """
        Computes feature importances based on the path lengths in the tree.

        Returns:
            feature_importances (array): The importance of each feature.
        """
        if self._root is None or self._num_features is None:
            raise ValueError("The tree has not been fitted yet.")
        importances = np.zeros(self._num_features)
        node_counts = np.zeros(self._num_features)
        self._accumulate_feature_importances(self._root, importances, node_counts)
        # Normalize importances
        importances /= np.sum(node_counts)
        return importances

    def _accumulate_feature_importances(
        self, node: Node, importances: np.ndarray, node_counts: np.ndarray
    ) -> None:
        """
        Helper method to accumulate feature importances from a single tree.

        Args:
            node (Node): The current node in the tree.
            importances (array): The array to accumulate importances.
            node_counts (array): The array to count occurrences of features.
        """
        if node.is_leaf() or (node.left is None or node.right is None):
            return
        # mini = max(1, min(node.left.size, node.right.size))
        mini = 1
        importances[node.feature] += 1 / mini
        node_counts[node.feature] += 1
        self._accumulate_feature_importances(node.left, importances, node_counts)
        self._accumulate_feature_importances(node.right, importances, node_counts)

    def fit(self, X) -> None:
        """
        Fits the IsolationTree to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        # create root node w/ subsample
        subsample = np.random.choice(X.shape[0], self.max_samples, replace=False)
        X = X[subsample]
        self.n_samples = X.shape[0]
        self._num_features = X.shape[1]
        self._root = self._fit_node(X, current_height=0)
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
            if self._root is None:
                raise ValueError("The tree has not been fitted yet.")
            node = self._root
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
        n_trees: int = 100,
        contamination: float | str = "auto",
        max_samples: int = 512,
        random_state: int | RandomState | None = None,
        verbose: bool = False,
        warm_start: bool = False,
    ):
        """
        Initializes the IsolationForest model.

        Args:
            n_trees (int): The number of base trees in the ensemble.
            contamination (float or 'auto'): The amount of contamination (proportion of outliers in the data set).
            max_samples (int): The number of samples to draw from X to train each base estimator.
            random_state (int or RandomState instance): Controls the randomness of the estimator.
            verbose (bool): Enable verbose output.
            warm_start (bool): When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.
        """
        # Initialize attributes based on parameters
        self.n_trees: int = n_trees

        self.contamination: float = (
            contamination if isinstance(contamination, float) else 0.1
        )
        self.max_samples: int = max_samples
        self._n_samples: int | None = None  # to be set during fit
        self.random_state: int | RandomState = (
            random_state if random_state is not None else RandomState()
        )
        self.verbose: bool = verbose
        self.warm_start: bool = warm_start
        self.estimators_: list[IsolationTree] = []

    def feature_importances_(self) -> np.ndarray:
        """
        Computes feature importances based on the average path lengths across all trees.

        Returns:
            feature_importances (array): The importance of each feature.
        """
        if not self.estimators_ or self._num_features is None:
            raise ValueError("The model has not been fitted yet.")
        importances = np.zeros(self._num_features)
        for estimator in self.estimators_:
            importances += estimator.feature_importances_()
        importances /= len(self.estimators_)
        return importances

    def fit(self, X):
        """
        Fits the IsolationForest model to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        self._n_samples = X.shape[0]
        self._num_features = X.shape[1]
        if not self.warm_start or not self.estimators_:
            self.estimators_ = [self._make_estimator() for _ in range(self.n_trees)]
            for estimator in self.estimators_:
                estimator.fit(X)
        else:
            new_estimators = [self._make_estimator() for _ in range(self.n_trees)]
            for estimator in new_estimators:
                estimator.fit(X)
            self.estimators_.extend(new_estimators)

    def _c(self, n: int) -> float:
        """
        Computes c(n): the average path length of unsuccessful searches in a binary search tree.
        Used for normalizing anomaly scores.

        Args:
            n (int): Number of samples in the dataset.

        Returns:
            float: The normalization constant.
        """
        if n <= 1:
            return 0.0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def _anomaly_score(self, x: np.ndarray) -> float:
        """
        Computes the normalized anomaly score for a single sample x.

        The formula is: s(x, n) = 2^(-E(h(x)) / c(n))
        where E(h(x)) is the average path length and c(n) is the normalization constant.

        Args:
            x (array-like): The input sample.
        Returns:
            score (float): The normalized anomaly score for the sample x.
        """
        if not self.estimators_ or self._n_samples is None:
            raise ValueError("The model has not been fitted yet.")
        # Get average path length across all trees
        avg_path_length = float(
            np.mean([estimator.trace_path(x) for estimator in self.estimators_])
        )
        # Normalize by c(n) and apply exponential formula
        c_n = self._c(self._n_samples)
        if c_n == 0:
            return 0.0
        return 2.0 ** (-(avg_path_length / c_n))

    def decision_function(self, X) -> np.ndarray:
        """
        Anomaly scores for X using the normalized isolation forest formula.

        Parameters:
            X (array-like): The input samples.

        Returns:
            scores (array): The normalized anomaly scores for each sample.
        """
        return np.array([self._anomaly_score(x) for x in X])

    def predict(self, X) -> np.ndarray:
        """
        Predicts if a particular sample is an outlier or not.

        Args:
            X (array-like): The input samples.

        Returns:
            predictions (array): Array of 1 for outliers and 0 for inliers.
        """
        scores = self.decision_function(X)
        threshold = np.percentile(scores, (100 * (1 - self.contamination)))
        return np.where(scores >= threshold, 1, 0)

    def fit_predict(self, X) -> np.ndarray:
        """
        Fits the model to the training data and returns the predictions.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        Returns:
            predictions (array): Array of 1 for outliers and 0 for inliers.
        """
        self.fit(X)
        return self.predict(X)

    def _make_estimator(self) -> IsolationTree:
        """
        Helper method to create a single Isolation Tree (base estimator).
        """
        if self._n_samples is None:
            raise ValueError("Something went wrong. _n_samples is not set.")
        return IsolationTree(
            max_samples=self.max_samples,
            random_state=self.random_state,
        )
