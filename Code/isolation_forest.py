from numpy.random import RandomState
import numpy as np


class IsolationTree:
    def __init__(
        self,
        max_samples: int,
        max_features: int,
        random_state: int | RandomState | None = None,
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
        self.random_state = random_state

    def fit(self, X, sample_weight=None) -> None:
        """
        Fits the IsolationTree to the training data.

        Args:
            X (array-like): The input samples.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        return None

    def path_length(self, X) -> np.ndarray:
        """
        Computes the path length for each sample in X.

        Args:
            X (array-like): The input samples.

        Returns:
            path_lengths (array): The path lengths for each sample.
        """
        return np.array([])


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

    def fit(self, X, y=None, sample_weight=None) -> None:
        """
        Fits the IsolationForest model to the training data.

        Args:
            X (array-like): The input samples.
            y (array-like, optional): Ignored. Not used in unsupervised learning.
            sample_weight (array-like, optional): Individual weights for each sample.

        """
        return None

    def decision_function(self, X) -> np.ndarray:
        """
        Average anomaly score of X of the base classifiers.

        Parameters:
            X (array-like): The input samples.

        Returns:
            scores (array): The anomaly scores for each sample.
        """
        return np.array([])

    def predict(self, X) -> np.ndarray:
        """
        Predicts if a particular sample is an outlier or not.

        Args:
            X (array-like): The input samples.

        Returns:
            predictions (array): Array of -1 for outliers and 1 for inliers.
        """
        return np.array([])

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

    def _make_estimator(self, random_state=None) -> IsolationTree:
        """
        Helper method to create a single Isolation Tree (base estimator).
        """
        return IsolationTree(
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=random_state,
        )
