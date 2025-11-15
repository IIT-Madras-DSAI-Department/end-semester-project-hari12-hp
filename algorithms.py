# algorithms.py
#
# This file contains all from-scratch model definitions for
# DA2401- Machine Learning Lab End Semester.
#
# code flow:
# Imports
# 2. Utility Functions & Classes
#    - calculate_f1_score
#    - StandardScaler
#    - PCA (Principal Component Analysis) for preprocessing of dataset
# 3. Wrapper Classes
#    - OneVsRestClassifier (Generic)
# 4. Core Classification Models
#    - LogisticRegression (Binary)
#    - KNearestNeighbors (Multiclass)
#    - DecisionTree (Multiclass, base for RF)
# 5. Ensemble Models
#    - RandomForest (Bagging, Multiclass)
#    - XGBoost (Boosting, Binary - from user)
#    - OneVsRestXGBoost (Wrapper for XGBoost)

import numpy as np
import pandas as pd
import time
from collections import Counter
import sys
import os
import copy # For the generic OneVsRestClassifier

# Suppress runtime warnings
np.seterr(over='ignore', invalid='ignore')

# 2. UTILITY FUNCTIONS & CLASSES


def calculate_f1_score(y_true, y_pred, average='macro'):
    """
    Calculates the F1-score without using sklearn.
    """
    classes = np.unique(y_true)
    f1_scores = []

    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    if average == 'macro':
        return np.mean(f1_scores)
    elif average == 'weighted':
        class_counts = Counter(y_true)
        class_f1_map = dict(zip(classes, f1_scores))
        total_samples = len(y_true)
        weighted_f1 = sum(class_counts[c] * class_f1_map[c] for c in classes) / total_samples
        return weighted_f1
    else: # 'micro' is just accuracy in a multiclass problem
        return np.mean(y_true == y_pred)


class StandardScaler:
    """
    Scales data to have a mean of 0 and a standard deviation of 1, this is important since pca is sensitive to scale.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Add epsilon to avoid division by zero
        self.std_[self.std_ == 0] = 1e-15 

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class PCA:
    """
    Principal Component Analysis (PCA) from scratch.
    """
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        # 1. Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. Compute covariance matrix
        cov_matrix = np.cov(X_centered.T)
        
        # 3. Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 4. Sort eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors.T # Transpose for easier indexing
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        # 5. Store the first n_components
        self.components_ = eigenvectors[:self.n_components]

    def transform(self, X):
        # Center the data
        X_centered = X - self.mean_
        
        # Project data onto the principal components
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# 3. WRAPPER CLASSES
class OneVsRestClassifier:
    """
    A generic One-vs-Rest (OvR) wrapper for any binary classifier
    that has .fit(), .predict_proba(), and .predict() methods.
    """
    def __init__(self, base_classifier):
        """
        base_classifier: An *instance* of a binary classifier.
                         e.g., LogisticRegression(learning_rate=0.1)
        """
        self.base_classifier = base_classifier
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            # Creates a binary target vector
            y_binary = (y == c).astype(int)
            
            # Creates a deep copy of the classifier instance
            model = copy.deepcopy(self.base_classifier)
            
            # Fit the model on the binary data
            model.fit(X, y_binary)
            self.models[c] = model

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        all_probas = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            model = self.models[c]
            # Get the probability of the positive class
            probas = model.predict_proba(X)
            all_probas[:, i] = probas
            
        return all_probas

    def predict(self, X):
        # Get the (n_samples, 10) probability matrix
        all_probas = self.predict_proba(X)
        
        # Find the index of the highest probability
        best_class_indices = np.argmax(all_probas, axis=1)
        
        # Map indices back to the actual class labels
        return self.classes_[best_class_indices]

# 4. CORE CLASSIFICATION MODELS
class LogisticRegression:
    """
    Binary Logistic Regression with Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        # We add random_state for reproducible weight initialization
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250) # Avoids overflow
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        # Small random values instead of zeros can help
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
        # Gradient Descent
        for _ in range(self.n_iterations):
            # Linear model: z = X.w + b
            z = np.dot(X, self.weights) + self.bias
            
            # Predictions (probabilities)
            y_pred = self._sigmoid(z)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)


class KNearestNeighbors:
    """
    K-Nearest Neighbors (KNN) Classifier.
    """
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN is a "lazy" learner, so fit just stores the data
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b)**2))

    def predict(self, X):
        predictions = [self._predict_one(x) for x in X]
        return np.array(predictions)

    def _predict_one(self, x):
        # 1. Calculate distances from x to all points in X_train
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # 2. Get the indices of the k-nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # 3. Get the labels of those neighbors
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        
        # 4. Return the most common class label (majority vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


class DecisionTree:
    """
    Decision Tree Classifier for multiclass classification.
    """
    # Helper class for storing node information
    class _TreeNode:
        def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value # This is the class label if it's a leaf
        
        def is_leaf_node(self):
            return self.value is not None

    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features # For Random Forest feature subsetting
        self.root = None

    def fit(self, X, y):
        # If n_features is set (by Random Forest), use a subset.
        # Otherwise, use all features.
        self.n_features = X.shape[1] if self.n_features is None else min(X.shape[1], self.n_features)
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return self._TreeNode(value=leaf_value)
            
        # Select a random subset of features
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        # Find the best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
        # If no split improves information gain, create a leaf
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return self._TreeNode(value=leaf_value)

        # Split the data
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = X[:, best_feat] > best_thresh
        
        # Ensure the split actually created two children
        if not np.any(left_idxs) or not np.any(right_idxs):
            leaf_value = self._most_common_label(y)
            return self._TreeNode(value=leaf_value)

        # Recursively build subtrees
        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return self._TreeNode(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        parent_gini = self._gini_impurity(y)
        n_samples = len(y)
        
        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                # Split
                left_idxs = X[:, feat_idx] <= thresh
                y_left, y_right = y[left_idxs], y[~left_idxs]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                    
                # Calculate weighted average gini
                n_l, n_r = len(y_left), len(y_right)
                gini_left, gini_right = self._gini_impurity(y_left), self._gini_impurity(y_right)
                child_gini = (n_l / n_samples) * gini_left + (n_r / n_samples) * gini_right
                
                # Information Gain
                gain = parent_gini - child_gini
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh
                    
        return split_idx, split_thresh

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        # Get counts of each class
        counts = np.bincount(y)
        # Calculate probabilities
        probabilities = counts[counts > 0] / len(y)
        # Gini = 1 - sum(p_i^2)
        return 1.0 - np.sum(probabilities**2)

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# 5. ENSEMBLE MODELS

class RandomForest:
    """
    Random Forest Classifier using Bagging.
    """
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, n_features=None, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state
        self.trees = []
        if self.random_state:
            np.random.seed(self.random_state)
            
    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        # Sample with replacement
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
        
    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def fit(self, X, y):
        self.trees = []
        
        # Set n_features to sqrt(total features) if not specified
        if self.n_features is None:
            self.n_features = int(np.sqrt(X.shape[1]))
            
        for _ in range(self.n_trees):
            # Create a tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            
            # Get a bootstrapped sample
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # Fit the tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Get predictions from all trees
        # Shape: (n_trees, n_samples)
        tree_preds = [tree.predict(X) for tree in self.trees]
        
        # Transpose to (n_samples, n_trees)
        tree_preds = np.array(tree_preds).T
        
        # Get the majority vote for each sample
        # (self._most_common_label is faster than scipy.stats.mode)
        y_pred = [self._most_common_label(preds) for preds in tree_preds]
        return np.array(y_pred)
        
    def predict_proba(self, X):
        """
        Predicts class probabilities for X.
        Returns a (n_samples, n_classes) array.
        """
        # This is a bit more complex, we need to know the classes
        # Let's assume classes are 0-9
        n_samples = X.shape[0]
        n_classes = 10 # Hardcoded for MNIST only
        all_probas = np.zeros((n_samples, n_classes))
        
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees]).T
        
        # For each sample, count the votes for each class
        for i in range(n_samples):
            votes = tree_preds[i]
            class_counts = Counter(votes)
            for c, count in class_counts.items():
                all_probas[i, c] = count / self.n_trees
                
        return all_probas

# 5. XGBOOST MODELS

def _sigmoid_xgb(z):
    z = np.clip(z, -30, 30); return 1 / (1 + np.exp(-z))
    
def _get_initial_prediction_xgb(y):
    p = np.mean(y); p = np.clip(p, 1e-15, 1 - 1e-15); return np.log(p / (1 - p))
    
def _get_derivatives_xgb(y_true, raw_predictions):
    p_hat = _sigmoid_xgb(raw_predictions); return p_hat - y_true, p_hat * (1 - p_hat)

class XGBDecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index, self.threshold, self.left, self.right, self.value = feature_index, threshold, left, right, value
    def is_leaf_node(self): return self.value is not None

class XGBDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, 
                 feature_indices=None, gamma=0.0, lambda_=1.0, n_bins=32):
        self.max_depth, self.min_samples_split, self.root = max_depth, min_samples_split, None
        self.feature_indices, self.gamma, self.lambda_ = feature_indices, gamma, lambda_
        self.n_bins = n_bins

    def fit(self, X, y_gh):
        if self.feature_indices is None: self.feature_indices = list(range(X.shape[1]))
        self.root = self._build_tree(X, y_gh)

    def _build_tree(self, X, y_gh, depth=0):
        n_samples = X.shape[0]
        if len(y_gh) == 0: return XGBDecisionTreeNode(value=0.0)
        if (depth >= self.max_depth or n_samples < self.min_samples_split):
            return XGBDecisionTreeNode(value=self._calculate_leaf_value(y_gh))

        best_feat, best_thresh, best_gain = self._best_split(X, y_gh, n_samples)

        if best_feat is None or best_gain <= 0:
            return XGBDecisionTreeNode(value=self._calculate_leaf_value(y_gh))
            
        left_idx, right_idx = X[:, best_feat] <= best_thresh, X[:, best_feat] > best_thresh
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return XGBDecisionTreeNode(value=self._calculate_leaf_value(y_gh))

        left = self._build_tree(X[left_idx], y_gh[left_idx], depth + 1)
        right = self._build_tree(X[right_idx], y_gh[right_idx], depth + 1)
        return XGBDecisionTreeNode(feature_index=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y_gh, n_samples):
        best_gain, split_idx, split_thresh = 0.0, None, None
        if n_samples == 0: return None, None, 0.0
            
        G_parent, H_parent = np.sum(y_gh[:, 0]), np.sum(y_gh[:, 1])
        score_P = G_parent**2 / (H_parent + self.lambda_ + 1e-6)

        for feat_idx in self.feature_indices:
            feature_values = X[:, feat_idx]
            min_val, max_val = np.min(feature_values), np.max(feature_values)
            if min_val == max_val: continue
            
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            bin_indices = np.digitize(feature_values, bin_edges[1:-1])

            G_bins = np.bincount(bin_indices, weights=y_gh[:, 0], minlength=self.n_bins + 1)
            H_bins = np.bincount(bin_indices, weights=y_gh[:, 1], minlength=self.n_bins + 1)
            count_bins = np.bincount(bin_indices, minlength=self.n_bins + 1)

            G_left, H_left, count_left = 0.0, 0.0, 0
            
            for i in range(self.n_bins):
                G_left += G_bins[i]
                H_left += H_bins[i]
                count_left += count_bins[i]

                if count_left < self.min_samples_split: continue
                
                G_right = G_parent - G_left
                H_right = H_parent - H_left
                count_right = n_samples - count_left

                if count_right < self.min_samples_split: continue
                
                thresh = bin_edges[i+1]
                
                score_L = G_left**2 / (H_left + self.lambda_ + 1e-6)
                score_R = G_right**2 / (H_right + self.lambda_ + 1e-6)
                gain = 0.5 * (score_L + score_R - score_P) - self.gamma

                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, feat_idx, thresh
                    
        return split_idx, split_thresh, best_gain

    def _calculate_leaf_value(self, y_gh):
        G, H = np.sum(y_gh[:, 0]), np.sum(y_gh[:, 1])
        return -G / (H + self.lambda_ + 1e-6)

    def predict(self, X):
        return np.array([self._predict(inputs, self.root) for inputs in X])

    def _predict(self, inputs, node):
        if node is None: return 0
        if node.is_leaf_node(): return node.value
        if inputs[node.feature_index] <= node.threshold:
            return self._predict(inputs, node.left)
        return self._predict(inputs, node.right)

class XGBoostClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, gamma=0.0, lambda_=1.0, 
                 subsample=1.0, colsample_bytree=1.0, random_state=None, n_bins=32):
        self.n_estimators, self.learning_rate, self.max_depth = n_estimators, learning_rate, max_depth
        self.min_samples_split, self.gamma, self.lambda_ = min_samples_split, gamma, lambda_
        self.subsample, self.colsample_bytree = subsample, colsample_bytree
        self.random_state, self.n_bins = random_state, n_bins
        self.trees, self.initial_prediction, self.training_time = [], None, 0.0
        if self.random_state: np.random.seed(self.random_state)

    def fit(self, X, y):
        start_time = time.time()
        n_samples, n_features = X.shape
        
        self.initial_prediction = _get_initial_prediction_xgb(y)
        raw_predictions = np.full(n_samples, self.initial_prediction, dtype=float)
        for i in range(self.n_estimators):
            
            g, h = _get_derivatives_xgb(y, raw_predictions)
            y_gh = np.c_[g, h]
            
            if self.subsample < 1.0:
                sample_indices = np.random.choice(n_samples, int(n_samples * self.subsample), replace=True)
                X_sub, y_gh_sub = X[sample_indices], y_gh[sample_indices]
            else:
                X_sub, y_gh_sub = X, y_gh

            if self.colsample_bytree < 1.0:
                feature_indices = np.random.choice(n_features, int(n_features * self.colsample_bytree), replace=False)
            else:
                feature_indices = list(range(n_features))

            tree = XGBDecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                feature_indices=feature_indices, gamma=self.gamma, lambda_=self.lambda_,
                n_bins=self.n_bins 
            )
            tree.fit(X_sub, y_gh_sub) 
            tree_preds = tree.predict(X)
            raw_predictions += self.learning_rate * tree_preds
            self.trees.append(tree)
            
        self.training_time = time.time() - start_time

    def predict_proba(self, X):
        raw_predictions = np.full(X.shape[0], self.initial_prediction, dtype=float)
        for tree in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X)
        return _sigmoid_xgb(raw_predictions)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
class OneVsRestXGBoost:
    def __init__(self, base_xgb_params):
        self.base_xgb_params = base_xgb_params
        self.models = {} 
        self.classes_ = None
        self.training_time = 0.0

    def fit(self, X, y):
        start_time = time.time()
        self.classes_ = np.unique(y)

        
        for i, class_label in enumerate(self.classes_):
            
            y_binary = (y == class_label).astype(int)
            model = XGBoostClassifier(**self.base_xgb_params)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                model.fit(X, y_binary)
            finally:
                sys.stdout.close()
                sys.stdout = old_stdout
            
            self.models[class_label] = model

        self.training_time = time.time() - start_time

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        all_probas = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.classes_):
            model = self.models[class_label]
            probas = model.predict_proba(X)
            all_probas[:, i] = probas
            
        return all_probas

    def predict(self, X):
        all_probas = self.predict_proba(X)
        best_class_indices = np.argmax(all_probas, axis=1)
        predictions = self.classes_[best_class_indices]
        return predictions