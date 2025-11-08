# Term Project — Isolation Forest on UCI Vertebral Column Dataset

> Milwaukee School of Engineering (MSOE)<br>
> Leigh Goetsch<br>
> CSC5601 - Theory of Machine Learning<br>
> Fall 2025

## Project Overview

This term project implements the Isolation Forest anomaly detection algorithm from scratch, then applies it to the [UCI Vertebral Column Dataset](https://shebuti.com/vertebral-dataset/) (found in `Data/vertebral.mat`).

## Course and Project Outcomes
- Understand the basic process of machine learning
- Understand the concepts of learning theory, i.e., what is learnable, bias, variance, overfitting
- Understand the concepts and application of supervised learning
- Analyze and implement basic machine learning algorithms
- Understand the role of optimization in machine learning
- The ability to assess the quality of predictions and inferences
- The ability to apply methods to real world data sets

## Dataset

- File: `Data/vertebral.mat`
- Source: [Vertebral Column dataset](https://shebuti.com/vertebral-dataset/) ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/212/vertebral+column))

Description: The original Vertebral Column dataset from the UCI Machine Learning Repository is a biomedical dataset created by Dr. Henrique da Mota (Group of Applied Research in Orthopaedics, Centre Médico-Chirurgical de Réadaptation des Massues, Lyon, France). Each instance (patient) is represented by six biomechanical attributes derived from the shape and orientation of the pelvis and lumbar spine (in this order): pelvic incidence, pelvic tilt, lumbar lordosis angle, sacral slope, pelvic radius, and grade of spondylolisthesis. The dataset convention uses two class labels: Normal (`NO`) and Abnormal (`AB`). For our anomaly detection experiments, `AB` is treated as the majority class (inliers) with 210 instances, while `NO` is downsampled from 100 to 30 instances and treated as outliers.

## Implementation Specs

- Inputs:
	- A numeric 2D array X of shape (n_samples, n_features).
	- Algorithm hyperparameters: number of trees (n_estimators), subsampling size (sample_size), maximum tree height (max_depth), random seed.
- Outputs:
	- An anomaly score per sample (higher score indicates more abnormal/outlier).
	- Predicted labels (outlier/inlier) using a chosen threshold.

## References

- Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. "Isolation forest." Proceedings of the 2008 Eighth IEEE International Conference on Data Mining. IEEE, 2008.
- scikit-learn `IsolationForest` docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
