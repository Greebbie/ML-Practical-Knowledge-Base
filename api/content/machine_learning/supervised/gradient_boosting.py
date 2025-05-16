def get_content():
    return {
        "section": [
            {
                "title": "Gradient Boosting: Core Concepts",
                "description": """
                <p>Gradient Boosting is an ensemble machine learning technique that combines multiple weak learners (typically decision trees) to create a strong predictive model. It builds models sequentially, with each new model attempting to correct the errors of the previous models.</p>
                <p>Key components:</p>
                <ul>
                    <li>Loss function to be optimized</li>
                    <li>Weak learner (base model) to make predictions</li>
                    <li>Additive model to add weak learners to minimize the loss function</li>
                    <li>Gradient descent procedure for minimization</li>
                </ul>
                """,
                "img": "img/gradient_boosting.png",
                "caption": "Visualization of sequential model building in gradient boosting."
            },
            {
                "title": "The Algorithm",
                "description": """
                <p>The gradient boosting algorithm works as follows:</p>
                <ol>
                    <li>Initialize the model with a constant value</li>
                    <li>For m = 1 to M (number of iterations):
                        <ul>
                            <li>Compute the negative gradient (pseudo-residuals) of the loss function</li>
                            <li>Fit a base learner (e.g., decision tree) to the pseudo-residuals</li>
                            <li>Compute multiplier (step size) using line search</li>
                            <li>Update the model by adding the new base learner, scaled by the multiplier</li>
                        </ul>
                    </li>
                    <li>Return the final model</li>
                </ol>
                """,
                "formula": """
                $$F_0(x) = \\arg\\min_{\\gamma} \\sum_{i=1}^{n} L(y_i, \\gamma)$$
                
                $$F_m(x) = F_{m-1}(x) + \\gamma_m h_m(x)$$
                
                $$\\text{where } h_m(x) \\text{ is fit to the negative gradient: } -\\left[\\frac{\\partial L(y_i, F_{m-1}(x_i))}{\\partial F_{m-1}(x_i)}\\right]$$
                """
            },
            {
                "title": "Common Loss Functions",
                "description": """
                <p>Different loss functions can be used depending on the task:</p>
                <ul>
                    <li><strong>L2 Loss (Mean Squared Error)</strong>: Used for regression tasks</li>
                    <li><strong>Binomial Log-Likelihood Loss</strong>: Used for binary classification</li>
                    <li><strong>Multinomial Log-Likelihood Loss</strong>: Used for multi-class classification</li>
                </ul>
                """,
                "formula": """
                $$\\text{L2 Loss: } L(y, F) = \\frac{1}{2}(y - F)^2$$
                
                $$\\text{Binary Log-Loss: } L(y, F) = -y \\log(p) - (1-y)\\log(1-p) \\text{ where } p = \\frac{1}{1+e^{-F}}$$
                """
            },
            {
                "title": "Key Hyperparameters",
                "description": """
                <p>Important hyperparameters in gradient boosting:</p>
                <ul>
                    <li><strong>Learning Rate (shrinkage)</strong>: Controls the contribution of each tree to the final outcome</li>
                    <li><strong>Number of Estimators</strong>: Total number of trees to build</li>
                    <li><strong>Max Depth</strong>: Maximum depth of each tree</li>
                    <li><strong>Subsampling</strong>: Fraction of samples to use for fitting individual base learners</li>
                    <li><strong>Feature Sampling</strong>: Fraction of features to consider when looking for the best split</li>
                </ul>
                <p>Smaller learning rates require more estimators but often yield better performance.</p>
                """
            }
        ],
        "implementation": """
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a gradient boosting regressor
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    random_state=42
)

gbr.fit(X_train, y_train)

# Make predictions
y_pred = gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Feature importance
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
plt.yticks(range(len(sorted_idx)), [f"Feature {i}" for i in sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Gradient Boosting Model')

# Visualize the learning curve (stage-wise performance)
test_errors = [mean_squared_error(y_test, pred) for pred in gbr.staged_predict(X_test)]
plt.figure()
plt.plot(range(1, len(test_errors) + 1), test_errors, label='Test MSE')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.title('Error vs. Number of Trees')
plt.legend()
""",
        "interview_examples": [
            {
                "title": "Gradient Boosting vs. Random Forests",
                "description": "How does Gradient Boosting differ from Random Forests, and what are the tradeoffs?",
                "code": """
# Gradient Boosting vs. Random Forests:

# 1. Sequential vs. Parallel:
#    - Gradient Boosting: Sequential (each tree corrects errors of previous trees)
#    - Random Forest: Parallel (each tree is built independently)

# 2. Error correction strategy:
#    - Gradient Boosting: Focuses on errors of previous models
#    - Random Forest: Focuses on introducing randomness to reduce overfitting

# 3. Training time:
#    - Gradient Boosting: Slower (sequential nature, cannot be parallelized easily)
#    - Random Forest: Faster (trees can be built in parallel)

# 4. Hyperparameter sensitivity:
#    - Gradient Boosting: More sensitive to hyperparameters
#    - Random Forest: More robust to hyperparameter settings

# 5. Performance vs. overfitting:
#    - Gradient Boosting: Often better performance but more prone to overfitting
#    - Random Forest: More resistant to overfitting but may have lower ceiling on performance

# Implementation comparison:
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time

# Same synthetic data as before
rf = RandomForestRegressor(n_estimators=100, random_state=42)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Measure training time
start = time.time()
rf.fit(X_train, y_train)
rf_time = time.time() - start

start = time.time()
gb.fit(X_train, y_train)
gb_time = time.time() - start

print(f"Random Forest training time: {rf_time:.2f}s")
print(f"Gradient Boosting training time: {gb_time:.2f}s")

# Compare performance
rf_mse = mean_squared_error(y_test, rf.predict(X_test))
gb_mse = mean_squared_error(y_test, gb.predict(X_test))

print(f"Random Forest MSE: {rf_mse:.4f}")
print(f"Gradient Boosting MSE: {gb_mse:.4f}")
"""
            },
            {
                "title": "Implementing Gradient Boosting from Scratch",
                "description": "How would you implement a simple gradient boosting algorithm from scratch for regression?",
                "code": """
# Simple gradient boosting implementation for regression (L2 loss)
from sklearn.tree import DecisionTreeRegressor

class SimpleGradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X, y):
        # Initialize with mean value
        self.initial_prediction = np.mean(y)
        F = np.ones(len(y)) * self.initial_prediction
        
        # Build trees sequentially
        for i in range(self.n_estimators):
            # Compute pseudo-residuals (negative gradient of L2 loss)
            residuals = y - F
            
            # Fit a regression tree to the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Make predictions with the tree
            update = tree.predict(X)
            
            # Update the model (with shrinkage)
            F += self.learning_rate * update
            
            # Store the tree
            self.trees.append(tree)
        
        return self
    
    def predict(self, X):
        # Start with initial prediction
        predictions = np.ones(len(X)) * self.initial_prediction
        
        # Add contributions from each tree
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
            
        return predictions

# Usage:
gbr_custom = SimpleGradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
gbr_custom.fit(X_train, y_train)
custom_preds = gbr_custom.predict(X_test)
custom_mse = mean_squared_error(y_test, custom_preds)
print(f"Custom Gradient Boosting MSE: {custom_mse:.4f}")
"""
            }
        ],
        "resources": [
            {"title": "Gradient Boosting Machines, a Tutorial", "url": "https://arxiv.org/abs/1912.02385"},
            {"title": "XGBoost: A Scalable Tree Boosting System", "url": "https://arxiv.org/abs/1603.02754"},
            {"title": "A Gentle Introduction to Gradient Boosting", "url": "https://explained.ai/gradient-boosting/"}
        ],
        "related_topics": [
            "Decision Trees", "Random Forests", "XGBoost", "LightGBM", "Ensemble Methods"
        ]
    } 