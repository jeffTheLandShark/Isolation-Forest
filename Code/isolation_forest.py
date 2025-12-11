from numpy.random import RandomState
import numpy as np
from typing import Optional


EPSILON = 1e-12
HARMONIC_CONST = 0.5772156649  # Euler-Mascheroni constant


def c_(n: int) -> float:
    """Expected path length c(n) for isolation trees (Liu et al., 2008)."""
    if n <= 1:
        return 0.0
    # Using approximation with harmonic number: H_{n-1} ≈ ln(n-1) + gamma + 1/(2(n-1)) ...
    return 2.0 * (np.log(n - 1) + HARMONIC_CONST) - 2.0 * (n - 1) / n


class Node:
    def __init__(
        self,
        feature: int = -1,
        split_value: float = 0.0,
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

    def _feature_importances(self) -> np.ndarray:
        """
        Computes feature importances based on the path lengths in the tree.

        Returns:
            feature_importances (array): The importance of each feature.
        """
        if self._root is None or self._num_features is None:
            raise ValueError("The tree has not been fitted yet.")
        importances = np.zeros(self._num_features, dtype=float)
        self._accumulate_feature_importances(self._root, importances)
        s = importances.sum()
        return importances if s == 0 else importances / s

    def _accumulate_feature_importances(
        self, node: Node, importances: np.ndarray
    ) -> None:
        """
        Helper method to accumulate feature importances from a single tree.

        Args:
            node (Node): The current node in the tree.
            importances (array): The array to accumulate importances.
        """
        if node is None or node.is_leaf() or node.left is None or node.right is None:
            return
        n = node.size
        nL = node.left.size
        nR = node.right.size
        # compute expected path-length reduction at this split
        parent_c = c_(n)
        child_c = (nL * c_(nL) + nR * c_(nR)) / n
        gain = parent_c - child_c
        if gain > 0 and 0 <= node.feature < importances.size:
            importances[node.feature] += gain
        self._accumulate_feature_importances(node.left, importances)
        self._accumulate_feature_importances(node.right, importances)

    def fit(self, X: np.ndarray) -> None:
        """
        Fits the IsolationTree to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D array (n_samples, n_features)")
        # safe subsample
        sample_size = min(self.max_samples, X.shape[0])
        if sample_size == X.shape[0]:
            subsampled_X = X.copy()
        else:
            idx = self._rng.choice(X.shape[0], size=sample_size, replace=False)
            subsampled_X = X[idx]
        self.n_samples = subsampled_X.shape[0]
        self._num_features = subsampled_X.shape[1]
        self._root = self._fit_node(subsampled_X, current_height=0)

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
        if (
            node.feature < 0
            or node.feature >= x.shape[0]
            and x.ndim == 1
            and x.size != 0
        ):
            raise ValueError("Invalid feature index during path tracing")
        if x[node.feature] < node.split_value:
            return self.trace_path(x, node.left, current_height + 1)
        else:
            return self.trace_path(x, node.right, current_height + 1)

    def path_length(self, X: np.ndarray) -> np.ndarray:
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
        n_trees: int = 500,
        contamination: float = 0.1,
        max_samples: int = 512,
        random_state: Optional[int | RandomState] = None,
        verbose: bool = False,
    ):
        """
        Initializes the IsolationForest model.

        Args:
            n_trees (int): The number of base trees in the ensemble.
            contamination (float): The amount of contamination (proportion of outliers in the data set).
            max_samples (int): The number of samples to draw from X to train each base estimator.
            random_state (int or RandomState instance): Controls the randomness of the estimator.
            verbose (bool): Enable verbose output.
        """
        if n_trees <= 0:
            raise ValueError("n_trees must be > 0")
        self.n_trees: int = n_trees
        self.comtamination: float = contamination
        self.max_samples: int = max_samples
        # forest-level RNG
        self._rng: RandomState = (
            random_state
            if isinstance(random_state, RandomState)
            else (
                RandomState(random_state) if random_state is not None else RandomState()
            )
        )
        self.verbose: bool = verbose
        self._estimators: list[IsolationTree] = []
        self._n_samples: Optional[int] = None
        self._num_features: Optional[int] = None

    def feature_importances_(self) -> np.ndarray:
        """
        Computes feature importances based on the average path lengths across all trees.

        Returns:
            feature_importances (array): The importance of each feature.
        """
        if not self._estimators or self._num_features is None:
            raise ValueError("The model has not been fitted yet.")
        importances = np.zeros(self._num_features, dtype=float)
        for estimator in self._estimators:
            importances += estimator._feature_importances()
        return importances / len(self._estimators)

    def fit(self, X: np.ndarray):
        """
        Fits the IsolationForest model to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        self._n_samples = X.shape[0]
        self._num_features = X.shape[1]

        new_estimators: list[IsolationTree] = []
        for _ in range(self.n_trees):
            # derive a seed per-tree so trees are independent but reproducible
            seed = int(self._rng.randint(0, 2**31 - 1))
            tree = self._make_estimator(seed=seed)
            tree.fit(X)
            new_estimators.append(tree)

        self._estimators = new_estimators

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Average anomaly score for each sample in X across all trees.

        The formula is: s(x, n) = 2^(-E(h(x)) / c(n))
        where E(h(x)) is the average path length and c(n) is the normalization constant.

        Args:
            X (array-like): The input samples.
        Returns:
            score (array): The anomaly scores for each sample.
        """
        if not self._estimators or self._n_samples is None:
            raise ValueError("Model not fitted")
        # Average path length per sample across trees
        avg_paths = np.array(
            [np.mean([tree.trace_path(x) for tree in self._estimators]) for x in X]
        )
        c_n = c_(self._n_samples)
        if c_n == 0:
            c_n = EPSILON
        return 2.0 ** (-(avg_paths / c_n))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts if a particular sample is an outlier or not.

        Args:
            X (array-like): The input samples.

        Returns:
            predictions (array): Array of 1 for outliers and 0 for inliers.
        """
        scores = self.decision_function(X)
        threshold = np.percentile(scores, 100.0 * (1.0 - self.comtamination))
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

    def _make_estimator(self, seed: Optional[int] = None) -> IsolationTree:
        return IsolationTree(max_samples=self.max_samples, random_state=seed)
