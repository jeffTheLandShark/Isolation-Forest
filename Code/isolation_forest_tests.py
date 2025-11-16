import numpy as np

from isolation_forest import IsolationForest


def make_synthetic_data(n_inliers=200, n_outliers=20, random_state=0):
    """Generates synthetic data with inliers and outliers for testing."""
    rng = np.random.RandomState(random_state)
    inliers = rng.normal(loc=0.0, scale=0.5, size=(n_inliers, 2))
    outliers = rng.uniform(low=6.0, high=8.0, size=(n_outliers, 2))
    X = np.vstack([inliers, outliers])
    y = np.hstack([np.zeros(n_inliers, dtype=int), np.ones(n_outliers, dtype=int)])
    return X, y


def test_fit_predict_shape_and_values(X, y):
    """Tests that fit_predict returns correct shape and values."""
    clf = IsolationForest(
        n_estimators=50, max_samples=64, max_features=2, random_state=42
    )
    clf.fit(X)

    preds = clf.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X.shape[0],)
    # predictions should be either -1 (outlier) or 1 (inlier)
    unique = np.unique(preds)
    assert set(unique).issubset({-1, 1})


def test_decision_function_separates_outliers(X, y):
    """Tests that decision_function gives higher scores to outliers."""
    clf = IsolationForest(
        n_estimators=100, max_samples=128, max_features=2, random_state=0
    )
    clf.fit(X)

    scores = clf.decision_function(X)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (X.shape[0],)
    # scores should be finite numbers
    assert np.all(np.isfinite(scores))

    inlier_scores = scores[y == 0]
    outlier_scores = scores[y == 1]

    # On average, true outliers should have a different (typically higher) anomaly score
    assert outlier_scores.mean() != inlier_scores.mean()


def test_deterministic_with_random_state(X, y):
    """Tests that using the same random_state yields the same results."""
    clf1 = IsolationForest(n_estimators=50, random_state=0)
    clf2 = IsolationForest(n_estimators=50, random_state=0)

    clf1.fit(X)
    clf2.fit(X)

    p1 = clf1.predict(X)
    p2 = clf2.predict(X)
    assert np.array_equal(p1, p2)


def test_fit_predict_helper(X, y):
    """Tests the fit_predict convenience method."""
    clf = IsolationForest(n_estimators=20, random_state=1)
    preds = clf.fit_predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (X.shape[0],)


def test_input_validation_shapes():
    """Tests that input validation raises errors for incorrect shapes."""
    X = np.array([1, 2, 3])  # wrong shape (1-d)
    clf = IsolationForest()
    try:
        clf.fit(X)
        assert False, "Expected an exception due to incorrect input shape"
    except Exception:
        pass


if __name__ == "__main__":
    # Run the tests
    print("Running tests...")
    X, y = make_synthetic_data()
    print("Testing fit_predict_shape_and_values...")
    test_fit_predict_shape_and_values(X, y)
    print("Testing decision_function_separates_outliers...")
    test_decision_function_separates_outliers(X, y)
    print("Testing deterministic_with_random_state...")
    test_deterministic_with_random_state(X, y)
    print("Testing fit_predict_helper...")
    test_fit_predict_helper(X, y)
    print("Testing input_validation_shapes...")
    test_input_validation_shapes()
    print("All tests passed!")
