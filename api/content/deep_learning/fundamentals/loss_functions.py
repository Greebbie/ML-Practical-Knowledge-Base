def get_content():
    return {
        "section": [
            {
                "title": "Loss Functions: Overview",
                "description": """
                <p>Loss functions (or cost functions) quantify the difference between predicted values and actual target values in a machine learning model. They provide a measure of how well the model is performing.</p>
                <p>Key characteristics of loss functions:</p>
                <ul>
                    <li>They must be differentiable for gradient-based optimization</li>
                    <li>They should accurately reflect the goals of the learning task</li>
                    <li>Different problems require different loss functions</li>
                    <li>They guide the optimization process during training</li>
                </ul>
                """
            },
            {
                "title": "Regression Loss Functions",
                "description": """
                <p>Regression loss functions are used when predicting continuous values.</p>
                <p>Common regression loss functions include:</p>
                <ul>
                    <li><strong>Mean Squared Error (MSE)</strong>: Penalizes larger errors quadratically</li>
                    <li><strong>Mean Absolute Error (MAE)</strong>: Measures the average absolute difference between predictions and targets</li>
                    <li><strong>Huber Loss</strong>: Combines MSE and MAE, less sensitive to outliers</li>
                    <li><strong>Log-cosh Loss</strong>: Smooth approximation of MSE, less affected by outliers</li>
                </ul>
                """,
                "formula": """
                $$\\text{MSE} = \\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2$$
                
                $$\\text{MAE} = \\frac{1}{n}\\sum_{i=1}^{n}|y_i - \\hat{y}_i|$$
                
                $$\\text{Huber Loss} = \\begin{cases}
                \\frac{1}{2}(y - \\hat{y})^2 & \\text{for } |y - \\hat{y}| \\leq \\delta \\\\
                \\delta|y - \\hat{y}| - \\frac{1}{2}\\delta^2 & \\text{otherwise}
                \\end{cases}$$
                """
            },
            {
                "title": "Classification Loss Functions",
                "description": """
                <p>Classification loss functions are used to measure the performance of classification models.</p>
                <p>Key classification losses include:</p>
                <ul>
                    <li><strong>Binary Cross-Entropy</strong>: For binary classification problems</li>
                    <li><strong>Categorical Cross-Entropy</strong>: For multi-class classification</li>
                    <li><strong>Sparse Categorical Cross-Entropy</strong>: When labels are integers rather than one-hot encoded</li>
                    <li><strong>Focal Loss</strong>: Modified cross-entropy that focuses on hard examples</li>
                    <li><strong>Hinge Loss</strong>: Used in SVMs and margin-based classifiers</li>
                </ul>
                """,
                "formula": """
                $$\\text{Binary Cross-Entropy} = -\\frac{1}{n}\\sum_{i=1}^{n}[y_i\\log(\\hat{y}_i) + (1-y_i)\\log(1-\\hat{y}_i)]$$
                
                $$\\text{Categorical Cross-Entropy} = -\\frac{1}{n}\\sum_{i=1}^{n}\\sum_{j=1}^{C}y_{ij}\\log(\\hat{y}_{ij})$$
                
                $$\\text{Hinge Loss} = \\frac{1}{n}\\sum_{i=1}^{n}\\max(0, 1 - y_i \\hat{y}_i)$$
                """
            },
            {
                "title": "Probabilistic Loss Functions",
                "description": """
                <p>Probabilistic loss functions treat the output of a model as parameters of a probability distribution. They are often derived from maximum likelihood estimation.</p>
                <p>Examples include:</p>
                <ul>
                    <li><strong>Kullback-Leibler Divergence</strong>: Measures how one probability distribution differs from another</li>
                    <li><strong>Negative Log-Likelihood</strong>: Derived from maximum likelihood estimation</li>
                    <li><strong>Maximum Mean Discrepancy</strong>: Measures the difference between two probability distributions</li>
                </ul>
                """,
                "formula": """
                $$\\text{KL Divergence} = \\sum_{i} p(x_i) \\log\\left(\\frac{p(x_i)}{q(x_i)}\\right)$$
                
                $$\\text{Negative Log-Likelihood} = -\\sum_{i=1}^{n}\\log(P(y_i|x_i; \\theta))$$
                """
            },
            {
                "title": "Advanced Loss Functions",
                "description": """
                <p>Advanced loss functions address specific challenges in deep learning.</p>
                <p>Notable examples include:</p>
                <ul>
                    <li><strong>Triplet Loss</strong>: Used in similarity learning and face recognition</li>
                    <li><strong>Contrastive Loss</strong>: For learning embeddings that group similar items together</li>
                    <li><strong>CTC Loss</strong>: Used in sequence prediction when alignments are unknown</li>
                    <li><strong>Dice Loss</strong>: For segmentation tasks, measuring overlap between predictions and targets</li>
                    <li><strong>Generative Adversarial Loss</strong>: Used in GANs for generator and discriminator training</li>
                </ul>
                """,
                "formula": """
                $$\\text{Triplet Loss} = \\sum_{i}^{n}\\max(||f(x_i^a) - f(x_i^p)||^2 - ||f(x_i^a) - f(x_i^n)||^2 + \\text{margin}, 0)$$
                
                $$\\text{Dice Loss} = 1 - \\frac{2|X \\cap Y|}{|X| + |Y|} = 1 - \\frac{2\\sum_{i}^{n}x_i y_i}{\\sum_{i}^{n}x_i^2 + \\sum_{i}^{n}y_i^2}$$
                """
            }
        ],
        "implementation": """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define some common loss functions in PyTorch
y_true = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
# Predictions in probability space (after sigmoid)
y_pred_prob = torch.tensor([0.9, 0.1, 0.8, 0.3, 0.7], dtype=torch.float32)
# Predictions in logit space (before sigmoid)
y_pred_logit = torch.tensor([2.2, -2.3, 1.3, -0.7, 0.8], dtype=torch.float32)

# Binary Cross-Entropy Loss
bce_loss = nn.BCELoss()
bce = bce_loss(y_pred_prob, y_true)
print(f"BCE Loss: {bce.item():.4f}")

# BCE with Logits (combines sigmoid and BCE, more stable)
bce_logits_loss = nn.BCEWithLogitsLoss()
bce_logits = bce_logits_loss(y_pred_logit, y_true)
print(f"BCE with Logits Loss: {bce_logits.item():.4f}")

# Multi-class example
# Class labels (batch_size=3, num_classes=4)
mc_true = torch.tensor([2, 0, 3])  # Class indices
mc_logits = torch.tensor([
    [-1.0, 0.2, 2.0, 0.5],  # Example 1 logits for classes 0,1,2,3
    [2.0, 0.1, 0.3, -0.2],  # Example 2 logits for classes 0,1,2,3
    [-0.3, 0.5, 0.2, 1.5]   # Example 3 logits for classes 0,1,2,3
])

# Cross-Entropy Loss (combines softmax and NLL)
ce_loss = nn.CrossEntropyLoss()
ce = ce_loss(mc_logits, mc_true)
print(f"Cross-Entropy Loss: {ce.item():.4f}")

# Visualize the behavior of different loss functions
def plot_loss_functions():
    # Generate true and predicted values
    y_true_val = 1.0
    y_pred_range = np.linspace(-2, 4, 1000)
    
    # Compute different losses
    mse_loss = (y_true_val - y_pred_range) ** 2
    mae_loss = np.abs(y_true_val - y_pred_range)
    
    # Huber loss with delta=1
    delta = 1.0
    huber_loss = np.where(
        np.abs(y_true_val - y_pred_range) < delta,
        0.5 * (y_true_val - y_pred_range) ** 2,
        delta * (np.abs(y_true_val - y_pred_range) - 0.5 * delta)
    )
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(y_pred_range, mse_loss, label='MSE')
    plt.plot(y_pred_range, mae_loss, label='MAE')
    plt.plot(y_pred_range, huber_loss, label='Huber (δ=1)')
    
    plt.axvline(x=y_true_val, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Prediction')
    plt.ylabel('Loss')
    plt.title('Comparison of Regression Loss Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot binary classification losses
    plt.figure(figsize=(12, 6))
    p = np.linspace(0.001, 0.999, 1000)  # avoid extremes for numerical stability
    
    # Binary class 1
    bce_loss_1 = -np.log(p)
    # Binary class 0
    bce_loss_0 = -np.log(1 - p)
    
    plt.plot(p, bce_loss_1, label='BCE (y=1)')
    plt.plot(p, bce_loss_0, label='BCE (y=0)')
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Loss')
    plt.title('Binary Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
plot_loss_functions()
""",
        "interview_examples": [
            {
                "title": "Choosing the Right Loss Function",
                "description": "How would you decide which loss function to use for a particular problem?",
                "code": """
# Guidelines for choosing loss functions:

# 1. Regression Tasks:
#    - MSE: General purpose, but sensitive to outliers
#    - MAE: More robust to outliers, but may converge slower
#    - Huber Loss: Good balance between MSE and MAE
#    - For heteroscedastic data: Consider weighted losses
#    - For very skewed distributions: Consider quantile losses

def choose_regression_loss(dataset_characteristics):
    if dataset_characteristics["has_outliers"]:
        if dataset_characteristics["training_stability_needed"]:
            return "Huber Loss"
        else:
            return "MAE"
    elif dataset_characteristics["needs_fast_convergence"]:
        return "MSE"
    elif dataset_characteristics["heteroscedastic"]:
        return "Weighted MSE"
    else:
        return "MSE"  # Default

# 2. Classification Tasks:
#    - Binary Classification: Binary Cross-Entropy
#    - Multi-class Classification: Categorical Cross-Entropy
#    - Imbalanced classes: Weighted Cross-Entropy or Focal Loss
#    - Ordinal Classification: Special ordinal losses

def choose_classification_loss(dataset_characteristics):
    if dataset_characteristics["num_classes"] == 2:
        if dataset_characteristics["imbalanced"]:
            if dataset_characteristics["highly_imbalanced"]:
                return "Focal Loss"
            else:
                return "Weighted BCE"
        else:
            return "BCE"
    else:  # Multi-class
        if dataset_characteristics["imbalanced"]:
            return "Weighted CE"
        else:
            return "Categorical CE"

# 3. Special Cases:
#    - Segmentation: Dice Loss, Focal Loss
#    - Object Detection: Combination of regression and classification losses
#    - GANs: Adversarial losses
#    - Metric Learning: Triplet Loss, Contrastive Loss
"""
            },
            {
                "title": "Implementing a Custom Loss Function in PyTorch",
                "description": "How would you implement a custom loss function in PyTorch?",
                "code": """
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Simple function
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary classification.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    """
    # Apply sigmoid to get probabilities
    p = torch.sigmoid(y_pred)
    
    # Calculate binary cross entropy
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')
    
    # Get the probabilities for the true class
    p_t = p * y_true + (1 - p) * (1 - y_true)
    
    # Apply the focusing parameter
    focal_weight = (1 - p_t) ** gamma
    
    # Apply the balancing parameter
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    
    # Compute the final loss
    loss = alpha_t * focal_weight * bce
    
    return loss.mean()

# Method 2: Using nn.Module (for more complex losses or those with parameters)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        """
        Dice Loss for segmentation.
        DL = 1 - (2*|X∩Y| + smooth) / (|X| + |Y| + smooth)
        """
        # Flatten predicted and true masks
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate intersection and unions
        intersection = (y_pred * y_true).sum()
        sum_preds = y_pred.sum()
        sum_targets = y_true.sum()
        
        # Calculate dice coefficient
        dice = (2. * intersection + self.smooth) / (sum_preds + sum_targets + self.smooth)
        
        # Return dice loss
        return 1 - dice

# Method 3: Custom autograd Function for more control
class CustomL1HingeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets, margin=1.0):
        ctx.save_for_backward(predictions, targets)
        ctx.margin = margin
        
        # Calculate loss: max(0, margin - pred*target)
        losses = torch.clamp(margin - predictions * targets, min=0)
        return losses.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        margin = ctx.margin
        
        # Gradient of the loss
        grad_input = torch.zeros_like(predictions)
        mask = (margin - predictions * targets) > 0
        grad_input[mask] = -targets[mask] * grad_output
        
        return grad_input, None, None  # Return gradients for each input

# Usage in a model
class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.l1 = nn.Linear(10, 5)
        self.dice_loss = DiceLoss(smooth=1.0)
        
    def forward(self, x, targets=None):
        x = self.l1(x)
        
        if targets is not None:
            # Calculate loss during training
            loss1 = focal_loss(x, targets, alpha=0.25, gamma=2.0)
            loss2 = self.dice_loss(x, targets)
            return x, loss1 + loss2
            
        return x  # Just return predictions during inference
"""
            }
        ],
        "resources": [
            {"title": "CS231n: Loss Functions and Optimization", "url": "http://cs231n.github.io/neural-networks-2/"},
            {"title": "Understanding Categorical Cross-Entropy Loss", "url": "https://gombru.github.io/2018/05/23/cross_entropy_loss/"},
            {"title": "A Gentle Introduction to Cross-Entropy for Machine Learning", "url": "https://machinelearningmastery.com/cross-entropy-for-machine-learning/"}
        ],
        "related_topics": [
            "Neural Networks", "Backpropagation", "Gradient Descent", "Optimization Algorithms", "Regularization"
        ]
    } 