## Implementation Notes

- Core libraries: NumPy, SciPy, pandas, matplotlib (for plotting). For comparisons only: scikit-learn.
- Implementation outline:
	1. Build isolation trees by recursively partitioning a subsample along a randomly chosen feature and split value until a stopping condition (single sample, zero variance, or max depth) is reached.
	2. For each sample, compute the path length from root to external node in each tree.
	3. Convert average path length to an anomaly score using the standard converting function (using the average path length for unsuccessful searches in binary trees as normalization).
	4. Aggregate scores across trees to form final anomaly score per sample.

Key implementation decisions to document:
- Subsampling strategy (random without replacement), recommended sample_size = 256 (or min(256, n_samples)).
- Split value selection: random between observed min and max on chosen feature, or random pick of two points and mid-point — document the chosen approach.
- Handling of identical feature values and zero-variance features.
- Use of recursion vs iterative tree construction (discuss memory/stack tradeoffs).
  
## Complexity Analysis (brief)

- Training time: O(n_trees * sample_size * log(sample_size) * d) in expectation, where d is number of features (expected tree height is O(log sample_size)).
- Prediction time: O(n_trees * height * n_samples_to_score) ≈ O(n_trees * log(sample_size) * n_samples).
- Space: O(n_trees * sample_size) to store tree nodes (each node stores split feature, split value, and child pointers) 
  - discuss memory for large n_trees or large sample_size.

## Experiments & Evaluation

Planned experiments:
- Run implementation and scikit-learn's IsolationForest on `vertebral.mat`.
- Metrics:
	- precision/recall/F1 and ROC AUC.
	- Compare running time and memory usage for different n_trees and sample_size.

Reproducibility notes:
- Fix random seeds for subsampling and split selection.
- Save model parameters and a small README with seeds used for experiments.


## Comparison to scikit-learn

Include an experiment that runs scikit-learn's `sklearn.ensemble.IsolationForest` with matched hyperparameters and compares:
- Anomaly score distribution
- Top-k flagged samples
- Running time for training and scoring



