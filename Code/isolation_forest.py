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
        max_features: int,
        n_samples: int,
        random_state: int | RandomState | None = None,
    ):
        """
        Initializes the IsolationTree.

        Args:
            max_features (int): The number of features to draw from X to train the tree.
            random_state (int or RandomState instance): Controls the randomness of the estimator.
        """
        self.max_features = max_features
        self.n_samples = n_samples
        self.random_state = random_state
        self.root_: Node | None = None
        self._set_random_state()

    def _set_random_state(self):
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif isinstance(self.random_state, RandomState):
            np.random.set_state(self.random_state.get_state())

    def _fit_node(self, X, current_height: int, max_height: int | None = None) -> Node:
        """
        Recursively fits a node in the IsolationTree.

        Args:
            X (array-like): The input samples.
            current_height (int): The current height of the node.
            max_height (int): The maximum height of the tree.

        """
        # if (current_height >= max_height) or (X.shape[0] <= 1) or np.all(X == X[0]):
        if (X.shape[0] <= 1) or np.all(X == X[0]):
            return Node(feature=-1, split_value=-1, size=X.shape[0])
        # random feature selection
        rand_feature = np.random.choice(self.max_features)
        feature = X[:, rand_feature]
        split_value = np.random.uniform(np.min(feature), np.max(feature))
        node = Node(feature=rand_feature, split_value=split_value, size=X.shape[0])
        # data partitioning
        left_partition, right_partition = node.partition(X)
        # recursion
        node.left = self._fit_node(left_partition, current_height + 1, max_height)
        node.right = self._fit_node(right_partition, current_height + 1, max_height)
        return node

    def feature_importances_(self) -> np.ndarray:
        """
        Computes feature importances based on the path lengths in the tree.

        Returns:
            feature_importances (array): The importance of each feature.
        """
        if self.root_ is None:
            raise ValueError("The tree has not been fitted yet.")
        importances = np.zeros(self.max_features)
        node_counts = np.zeros(self.max_features)
        self._accumulate_feature_importances(self.root_, importances, node_counts)
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

    def fit(self, X, sample_weight=None) -> None:
        """
        Fits the IsolationTree to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        # create root node w/ subsample
        self.root_ = self._fit_node(X, current_height=0)
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
        n_trees: int = 100,
        contamination: float | str = "auto",
        random_state: int | RandomState | None = None,
        verbose: bool = False,
        warm_start: bool = False,
    ):
        """
        Initializes the IsolationForest model.

        Args:
            n_trees (int): The number of base trees in the ensemble.
            contamination (float or 'auto'): The amount of contamination (proportion of outliers in the data set).
            max_features (int): The number of features to draw from X to train each base estimator.
            random_state (int or RandomState instance): Controls the randomness of the estimator.
            verbose (bool): Enable verbose output.
            warm_start (bool): When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.
        """
        # Initialize attributes based on parameters
        self.n_trees: int = n_trees

        self.contamination: float = (
            contamination if isinstance(contamination, float) else 0.1
        )
        self.num_features: int | None = None  # to be set during fit
        self.n_samples_: int | None = None  # to be set during fit
        self.random_state: int | RandomState = (
            random_state if random_state is not None else RandomState()
        )
        self.verbose: bool = verbose
        self.warm_start: bool = warm_start
        self.estimators_: list[IsolationTree] = []

    def shortcut_feature_importances_(self, X: np.ndarray) -> np.ndarray:
        """
        Computes feature importances based on the average path lengths across all trees.

        Returns:
            feature_importances (array): The importance of each feature.
        """
        if not self.estimators_ or self.num_features is None:
            raise ValueError("The model has not been fitted yet.")
        importances = np.zeros(self.num_features)
        for estimator in self.estimators_:
            importances += (
                estimator.feature_importances_() / estimator.path_length(X).mean()
            )
        importances /= len(self.estimators_)
        return importances

    def feature_importances_(self) -> np.ndarray:
        """
        Computes feature importances based on the average path lengths across all trees.

        Returns:
            feature_importances (array): The importance of each feature.
        """
        if not self.estimators_ or self.num_features is None:
            raise ValueError("The model has not been fitted yet.")
        importances = np.zeros(self.num_features)
        for estimator in self.estimators_:
            importances += estimator.feature_importances_()
        importances /= len(self.estimators_)
        return importances

    def fit(self, X, sample_weight=None):
        """
        Fits the IsolationForest model to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        self.num_features = X.shape[1]
        self.n_samples_ = X.shape[0]
        if not self.warm_start or not self.estimators_:
            self.estimators_ = [self._make_estimator() for _ in range(self.n_trees)]
            for estimator in self.estimators_:
                estimator.fit(X, sample_weight=sample_weight)
        else:
            new_estimators = [self._make_estimator() for _ in range(self.n_trees)]
            for estimator in new_estimators:
                estimator.fit(X, sample_weight=sample_weight)
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
        if not self.estimators_ or self.n_samples_ is None:
            raise ValueError("The model has not been fitted yet.")
        # Get average path length across all trees
        avg_path_length = float(
            np.mean([estimator.trace_path(x) for estimator in self.estimators_])
        )
        # Normalize by c(n) and apply exponential formula
        # Note: We need to store n_samples during fit
        c_n = self._c(self.n_samples_)
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

    def fit_predict(self, X, sample_weight=None) -> np.ndarray:
        """
        Fits the model to the training data and returns the predictions.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        Returns:
            predictions (array): Array of 1 for outliers and 0 for inliers.
        """
        self.fit(X, sample_weight=sample_weight)
        return self.predict(X)

    def _make_estimator(self) -> IsolationTree:
        """
        Helper method to create a single Isolation Tree (base estimator).
        """
        if self.num_features is None:
            raise ValueError("num_features must be set before creating an estimator.")
        if self.n_samples_ is None:
            raise ValueError("n_samples_ must be set before creating an estimator.")
        return IsolationTree(
            max_features=self.num_features,
            n_samples=self.n_samples_,
            random_state=self.random_state,
        )
