"""
Comparison script between custom Isolation Forest implementation and scikit-learn's.

This script compares:
- Anomaly score distributions
- Top-k flagged samples
- Running time for training and scoring
- Precision, recall, F1 on datasets with known anomalies
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from isolation_forest import IsolationForest


class IsolationForestComparison:
    """Comparison class for custom and sklearn Isolation Forest implementations."""

    def __init__(
        self,
        custom_model: IsolationForest,
        sklearn_model: SklearnIsolationForest,
        X_train,
        X_test=None,
        y_test=None,
        contamination=0.1,
    ):
        """
        Initialize comparison objects.

        Args:
            custom_model: Custom IsolationForest instance
            sklearn_model: scikit-learn IsolationForest instance
            X_train: Training data
            X_test: Test data (optional)
            y_test: Ground truth labels (optional, 1 = anomaly, 0 = normal)
            contamination: Contamination level
        """
        self.custom_model = custom_model
        self.sklearn_model = sklearn_model
        self.X_train = X_train
        self.X_test = X_test if X_test is not None else X_train
        self.y_test = y_test
        self.contamination = contamination
        self.results = {}

    def compare_training_time(self):
        """Compare training time between implementations."""
        print("\n" + "=" * 60)
        print("TRAINING TIME COMPARISON")
        print("=" * 60)

        # Custom implementation
        start = time.time()
        self.custom_model.fit(self.X_train)
        custom_time = time.time() - start

        # Scikit-learn implementation
        start = time.time()
        self.sklearn_model.fit(self.X_train)
        sklearn_time = time.time() - start

        print(f"Custom Implementation:  {custom_time:.4f} seconds")
        print(f"Scikit-learn:           {sklearn_time:.4f} seconds")
        print(f"Ratio (sklearn/custom): {sklearn_time / custom_time:.2f}x")

        self.results["training_time"] = {
            "custom": custom_time,
            "sklearn": sklearn_time,
        }
        return custom_time, sklearn_time

    def compare_scoring_time(self):
        """Compare scoring time between implementations."""
        print("\n" + "=" * 60)
        print("SCORING TIME COMPARISON")
        print("=" * 60)

        # Custom implementation
        start = time.time()
        custom_scores = self.custom_model.decision_function(self.X_test)
        custom_time = time.time() - start

        # Scikit-learn implementation
        start = time.time()
        sklearn_scores = self.sklearn_model.decision_function(self.X_test)
        sklearn_time = time.time() - start

        print(f"Custom Implementation:  {custom_time:.4f} seconds")
        print(f"Scikit-learn:           {sklearn_time:.4f} seconds")
        print(f"Ratio (sklearn/custom): {sklearn_time / custom_time:.2f}x")

        self.results["scoring_time"] = {
            "custom": custom_time,
            "sklearn": sklearn_time,
        }
        return custom_scores, sklearn_scores, custom_time, sklearn_time

    def compare_predictions(self):
        """Compare predictions between implementations."""
        print("\n" + "=" * 60)
        print("PREDICTION COMPARISON")
        print("=" * 60)

        custom_preds = self.custom_model.predict(self.X_test)
        sklearn_preds = self.sklearn_model.predict(self.X_test)

        # sklearn returns -1 for anomalies, 1 for normal
        # custom returns 1 for anomalies, 0 for normal
        # Convert for comparison
        sklearn_preds_converted = np.where(sklearn_preds == -1, 1, 0)

        agreement = np.mean(custom_preds == sklearn_preds_converted)
        print(f"\nPrediction Agreement: {agreement:.2%}")

        custom_anomalies = np.sum(custom_preds == 1)
        sklearn_anomalies = np.sum(sklearn_preds_converted == 1)

        print(
            f"Custom detected anomalies:    {custom_anomalies} ({100*custom_anomalies/len(custom_preds):.2f}%)"
        )
        print(
            f"Scikit-learn detected anomalies: {sklearn_anomalies} ({100*sklearn_anomalies/len(sklearn_preds_converted):.2f}%)"
        )

        self.results["predictions"] = {
            "custom": custom_preds,
            "sklearn": sklearn_preds_converted,
            "agreement": agreement,
        }
        return custom_preds, sklearn_preds_converted

    def compare_scores(self):
        """Compare anomaly scores between implementations."""
        print("\n" + "=" * 60)
        print("ANOMALY SCORE COMPARISON")
        print("=" * 60)

        custom_scores = self.custom_model.decision_function(self.X_test)
        sklearn_scores = self.sklearn_model.decision_function(self.X_test)

        print(f"\nCustom scores:")
        print(f"  Mean: {np.mean(custom_scores):.4f}")
        print(f"  Std:  {np.std(custom_scores):.4f}")
        print(f"  Min:  {np.min(custom_scores):.4f}")
        print(f"  Max:  {np.max(custom_scores):.4f}")

        print(f"\nScikit-learn scores:")
        print(f"  Mean: {np.mean(sklearn_scores):.4f}")
        print(f"  Std:  {np.std(sklearn_scores):.4f}")
        print(f"  Min:  {np.min(sklearn_scores):.4f}")
        print(f"  Max:  {np.max(sklearn_scores):.4f}")

        # Correlation between scores
        correlation = np.corrcoef(custom_scores, sklearn_scores)[0, 1]
        print(f"\nScore Correlation: {correlation:.4f}")

        self.results["scores"] = {
            "custom": custom_scores,
            "sklearn": sklearn_scores,
            "correlation": correlation,
        }
        return custom_scores, sklearn_scores

    def compare_top_anomalies(self, k=10):
        """Compare top-k flagged anomalies between implementations."""
        print("\n" + "=" * 60)
        print(f"TOP {k} ANOMALIES COMPARISON")
        print("=" * 60)

        custom_scores = self.custom_model.decision_function(self.X_test)
        sklearn_scores = self.sklearn_model.decision_function(self.X_test)

        custom_top_k = np.argsort(custom_scores)[-k:][::-1]
        sklearn_top_k = np.argsort(sklearn_scores)[-k:][::-1]

        overlap = len(np.intersect1d(custom_top_k, sklearn_top_k))
        print(f"\nOverlap in top {k}: {overlap}/{k} samples ({100*overlap/k:.1f}%)")

        self.results["top_anomalies"] = {
            "custom_indices": custom_top_k,
            "sklearn_indices": sklearn_top_k,
            "overlap": overlap,
        }
        return custom_top_k, sklearn_top_k

    def evaluate_with_labels(self):
        """Evaluate both models if ground truth labels are available."""
        if self.y_test is None:
            print("\nNo ground truth labels available for evaluation.")
            return None

        print("\n" + "=" * 60)
        print("EVALUATION WITH GROUND TRUTH")
        print("=" * 60)

        custom_preds = self.custom_model.predict(self.X_test)
        sklearn_preds = self.sklearn_model.predict(self.X_test)
        sklearn_preds = np.where(sklearn_preds == -1, 1, 0)

        # Convert labels if needed (assuming 0=normal, 1=anomaly in y_test)
        custom_precision = precision_score(self.y_test, custom_preds, zero_division=0)
        custom_recall = recall_score(self.y_test, custom_preds, zero_division=0)
        custom_f1 = f1_score(self.y_test, custom_preds, zero_division=0)

        sklearn_precision = precision_score(self.y_test, sklearn_preds, zero_division=0)
        sklearn_recall = recall_score(self.y_test, sklearn_preds, zero_division=0)
        sklearn_f1 = f1_score(self.y_test, sklearn_preds, zero_division=0)

        print(f"\nCustom Implementation:")
        print(f"  Precision: {custom_precision:.4f}")
        print(f"  Recall:    {custom_recall:.4f}")
        print(f"  F1 Score:  {custom_f1:.4f}")

        print(f"\nScikit-learn:")
        print(f"  Precision: {sklearn_precision:.4f}")
        print(f"  Recall:    {sklearn_recall:.4f}")
        print(f"  F1 Score:  {sklearn_f1:.4f}")

        # ROC AUC
        custom_scores = self.custom_model.decision_function(self.X_test)
        sklearn_scores = self.sklearn_model.decision_function(self.X_test)

        # Normalize scores to [0, 1] for ROC AUC
        custom_scores_norm = (custom_scores - np.min(custom_scores)) / (
            np.max(custom_scores) - np.min(custom_scores)
        )
        sklearn_scores_norm = (sklearn_scores - np.min(sklearn_scores)) / (
            np.max(sklearn_scores) - np.min(sklearn_scores)
        )

        try:
            custom_auc = roc_auc_score(self.y_test, custom_scores_norm)
            sklearn_auc = roc_auc_score(self.y_test, sklearn_scores_norm)

            print(f"\nROC AUC:")
            print(f"  Custom:     {custom_auc:.4f}")
            print(f"  Scikit-learn: {sklearn_auc:.4f}")

            self.results["evaluation"] = {
                "custom": {
                    "precision": custom_precision,
                    "recall": custom_recall,
                    "f1": custom_f1,
                    "auc": custom_auc,
                },
                "sklearn": {
                    "precision": sklearn_precision,
                    "recall": sklearn_recall,
                    "f1": sklearn_f1,
                    "auc": sklearn_auc,
                },
            }
        except Exception as e:
            print(f"\nCould not compute ROC AUC: {e}")

    def plot_score_distributions(self, save_path=None):
        """Plot distributions of anomaly scores."""
        custom_scores = self.custom_model.decision_function(self.X_test)
        sklearn_scores = self.sklearn_model.decision_function(self.X_test)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Custom scores
        axes[0].hist(custom_scores, bins=50, alpha=0.7, color="blue", edgecolor="black")
        axes[0].set_title(
            "Custom Implementation - Anomaly Scores", fontsize=12, fontweight="bold"
        )
        axes[0].set_xlabel("Anomaly Score")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(alpha=0.3)

        # Sklearn scores
        axes[1].hist(
            sklearn_scores, bins=50, alpha=0.7, color="green", edgecolor="black"
        )
        axes[1].set_title(
            "Scikit-learn - Anomaly Scores", fontsize=12, fontweight="bold"
        )
        axes[1].set_xlabel("Anomaly Score")
        axes[1].set_ylabel("Frequency")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def plot_score_comparison_scatter(self, save_path=None):
        """Scatter plot comparing scores from both implementations."""
        custom_scores = self.custom_model.decision_function(self.X_test)
        sklearn_scores = self.sklearn_model.decision_function(self.X_test)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(custom_scores, sklearn_scores, alpha=0.5, s=20)

        # Add diagonal line
        min_val = min(np.min(custom_scores), np.min(sklearn_scores))
        max_val = max(np.max(custom_scores), np.max(sklearn_scores))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Agreement",
        )

        ax.set_xlabel("Custom Implementation Score", fontsize=11)
        ax.set_ylabel("Scikit-learn Score", fontsize=11)
        ax.set_title("Anomaly Score Comparison", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def generate_summary_report(self):
        """Generate a summary report of all comparisons."""
        print("\n" + "=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)

        if "training_time" in self.results:
            custom_t = self.results["training_time"]["custom"]
            sklearn_t = self.results["training_time"]["sklearn"]
            print(f"\nTraining Time (seconds):")
            print(f"  Custom:     {custom_t:.4f}")
            print(f"  Scikit-learn: {sklearn_t:.4f}")

        if "scoring_time" in self.results:
            custom_t = self.results["scoring_time"]["custom"]
            sklearn_t = self.results["scoring_time"]["sklearn"]
            print(f"\nScoring Time (seconds):")
            print(f"  Custom:     {custom_t:.4f}")
            print(f"  Scikit-learn: {sklearn_t:.4f}")

        if "scores" in self.results:
            corr = self.results["scores"]["correlation"]
            print(f"\nAnomaly Score Correlation: {corr:.4f}")

        if "predictions" in self.results:
            agree = self.results["predictions"]["agreement"]
            print(f"Prediction Agreement: {agree:.2%}")

        if "top_anomalies" in self.results:
            overlap = self.results["top_anomalies"]["overlap"]
            print(f"Top 10 Anomaly Overlap: {overlap}/10")


def load_data(data_path, data_type="csv"):
    """Load data from file."""
    df_data = pd.read_csv(data_path)

    # drop cols with identical values
    nunique = df_data.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df_data = df_data.drop(columns=cols_to_drop)
    # drop nulls
    df_data = df_data.dropna()
    df_data["target"] = np.where(df_data["target"] == 1, 0, 1)  # make anomalies = 1

    X = df_data.drop(columns=["target"]).values
    y = df_data["target"].values
    return X, y
    if data_type == "csv":
        df = pd.read_csv(data_path)
        X = df.iloc[:, :-1].values if df.shape[1] > 1 else df.values
        y = df.iloc[:, -1].values if df.shape[1] > 1 else None

    elif data_type == "mat":
        mat_data = loadmat(data_path)
        # Assuming the data is stored in a key like 'X' or similar
        key = [k for k in mat_data.keys() if not k.startswith("__")][0]
        data = mat_data[key]
        if len(data.shape) == 2 and data.shape[1] > 1:
            X = data[:, :-1]
            y = data[:, -1]
        else:
            X = data
            y = None
    return X, y


def main():
    """Main comparison function."""
    # Example usage with a dataset
    print("Isolation Forest Implementation Comparison")
    print("Custom vs Scikit-learn")

    # Load data (example: vertebral.mat)
    data_path = "/home/ad.msoe.edu/goetschm/CSC5601/term_proj/Isolation-Forest/Data/TUANDROMD.csv"
    print(f"\nLoading data from {data_path}...")

    try:
        X, y = load_data(data_path, data_type="csv")
        print(f"Data shape: {X.shape}")
        print(f"Labels shape: {y.shape if y is not None else 'None'}")
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please ensure the data file exists in the Data directory.")

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Set parameters
    n_trees = 100
    contamination = 0.1
    random_state = 42

    print(f"\nParameters:")
    print(f"  n_trees: {n_trees}")
    print(f"  contamination: {contamination}")
    print(f"  random_state: {random_state}")

    # Initialize models
    custom_model = IsolationForest(
        n_trees=n_trees,
        contamination=contamination,
        random_state=random_state,
    )

    sklearn_model = SklearnIsolationForest(
        n_estimators=n_trees,
        max_samples=512,
        contamination=contamination,
        random_state=random_state,
    )

    # Create comparison object
    comparison = IsolationForestComparison(
        custom_model=custom_model,
        sklearn_model=sklearn_model,
        X_train=X,
        X_test=X,
        y_test=y,
        contamination=contamination,
    )

    # Run comparisons
    comparison.compare_training_time()
    comparison.compare_scoring_time()
    comparison.compare_scores()
    comparison.compare_predictions()
    comparison.compare_top_anomalies(k=10)

    if y is not None:
        comparison.evaluate_with_labels()

    # Generate report
    comparison.generate_summary_report()

    try:
        # Create visualizations
        print("\nGenerating visualizations...")
        comparison.plot_score_distributions("./outputs/score_distributions.png")
        comparison.plot_score_comparison_scatter("./outputs/score_comparison.png")
        print("Plots saved!")
    except FileNotFoundError as e:
        print(f"Error saving plots: {e}")
        print("Please ensure the specified save paths are valid.")


if __name__ == "__main__":
    main()
