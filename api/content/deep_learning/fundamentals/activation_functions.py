def get_content():
    return {
        "section": [
            {
                "title": "Activation Functions: Overview",
                "description": """
                <p>Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns. Without activation functions, neural networks would only be capable of learning linear relationships.</p>
                <p>Key properties of activation functions:</p>
                <ul>
                    <li>Non-linearity</li>
                    <li>Differentiability (for backpropagation)</li>
                    <li>Range and monotonicity</li>
                    <li>Computational efficiency</li>
                </ul>
                """,
                "img": "img/activation_comparison.png",
                "caption": "Comparison of common activation functions."
            },
            {
                "title": "Sigmoid Function",
                "description": """
                <p>The sigmoid function squashes input values to the range (0, 1). It was historically popular but has fallen out of favor due to problems with vanishing gradients and non-zero centered output.</p>
                <p>Properties:</p>
                <ul>
                    <li>Output range: (0, 1)</li>
                    <li>Smooth gradient</li>
                    <li>Suffers from vanishing gradient for extreme inputs</li>
                    <li>Output not zero-centered</li>
                </ul>
                """,
                "formula": "$$\\sigma(x) = \\frac{1}{1 + e^{-x}}$$"
            },
            {
                "title": "Hyperbolic Tangent (tanh)",
                "description": """
                <p>The tanh function is similar to sigmoid but maps values to the range (-1, 1), making it zero-centered. This helps with the training dynamics of neural networks.</p>
                <p>Properties:</p>
                <ul>
                    <li>Output range: (-1, 1)</li>
                    <li>Zero-centered output</li>
                    <li>Still suffers from vanishing gradient for extreme inputs</li>
                </ul>
                """,
                "formula": "$$\\tanh(x) = \\frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\\sigma(2x) - 1$$"
            },
            {
                "title": "Rectified Linear Unit (ReLU)",
                "description": """
                <p>ReLU is currently the most widely used activation function due to its computational efficiency and ability to mitigate the vanishing gradient problem.</p>
                <p>Properties:</p>
                <ul>
                    <li>Output range: [0, ∞)</li>
                    <li>Computationally efficient</li>
                    <li>Helps mitigate the vanishing gradient problem</li>
                    <li>Non-differentiable at x=0</li>
                    <li>Suffers from "dying ReLU" problem (neurons can permanently die when large gradients flow through)</li>
                </ul>
                """,
                "formula": "$$\\text{ReLU}(x) = \\max(0, x) = \\begin{cases} x & \\text{if } x > 0 \\\\ 0 & \\text{if } x \\leq 0 \\end{cases}$$"
            },
            {
                "title": "Leaky ReLU and Variants",
                "description": """
                <p>Leaky ReLU and its variants address the dying ReLU problem by allowing a small, non-zero gradient when the unit is not active.</p>
                <p>Variants include:</p>
                <ul>
                    <li>Leaky ReLU: Uses a small fixed slope for negative inputs</li>
                    <li>Parametric ReLU (PReLU): Learns the slope parameter during training</li>
                    <li>Exponential Linear Unit (ELU): Smooths the function with exponential behavior for negative inputs</li>
                    <li>Scaled Exponential Linear Unit (SELU): Self-normalizes activations</li>
                </ul>
                """,
                "formula": """
                $$\\text{Leaky ReLU}(x) = \\begin{cases} x & \\text{if } x > 0 \\\\ \\alpha x & \\text{if } x \\leq 0 \\end{cases}$$
                
                $$\\text{ELU}(x) = \\begin{cases} x & \\text{if } x > 0 \\\\ \\alpha(e^x - 1) & \\text{if } x \\leq 0 \\end{cases}$$
                """
            },
            {
                "title": "Softmax Function",
                "description": """
                <p>The softmax function normalizes an N-dimensional vector of arbitrary real values to a probability distribution. It's commonly used in the output layer of classification networks.</p>
                <p>Properties:</p>
                <ul>
                    <li>Outputs sum to 1, representing a probability distribution</li>
                    <li>Emphasizes the largest values while suppressing significantly smaller ones</li>
                    <li>Applied to the final layer for multi-class classification problems</li>
                </ul>
                """,
                "formula": "$$\\text{softmax}(x_i) = \\frac{e^{x_i}}{\\sum_{j=1}^{n} e^{x_j}}$$"
            }
        ],
        "implementation": """
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

# Plot activation functions
x = np.linspace(-5, 5, 1000)
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU (α=0.01)')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, elu(x))
plt.title('ELU (α=1.0)')
plt.grid(True)

# Plot derivatives
plt.subplot(2, 3, 6)
plt.plot(x, sigmoid(x) * (1 - sigmoid(x)), label='Sigmoid')
plt.plot(x, 1 - tanh(x)**2, label='Tanh')
plt.plot(x, np.where(x > 0, 1, 0), label='ReLU')
plt.title('Derivatives')
plt.legend()
plt.grid(True)

plt.tight_layout()
""",
        "interview_examples": [
            {
                "title": "Comparing Activation Functions",
                "description": "Explain the trade-offs between different activation functions and when to use each.",
                "code": """
# Activation function comparison:

# 1. Sigmoid:
#    - Pros: Smooth gradient, output bounded between 0 and 1
#    - Cons: Vanishing gradient, not zero-centered, computationally expensive
#    - Use case: Output layer for binary classification

# 2. Tanh:
#    - Pros: Zero-centered, output bounded between -1 and 1
#    - Cons: Still suffers from vanishing gradient
#    - Use case: Hidden layers when zero-centered output is important

# 3. ReLU:
#    - Pros: Computationally efficient, mitigates vanishing gradient
#    - Cons: "Dying ReLU" problem, not zero-centered
#    - Use case: Default choice for hidden layers in CNNs and many other networks

# 4. Leaky ReLU:
#    - Pros: Fixes dying ReLU problem, all benefits of ReLU
#    - Cons: Additional hyperparameter (leak coefficient)
#    - Use case: When dying neurons are a concern

# 5. ELU:
#    - Pros: Smooth function, negative outputs can push mean activation closer to zero
#    - Cons: Computationally more expensive than ReLU
#    - Use case: When slightly better accuracy than ReLU is needed

# 6. Softmax:
#    - Pros: Outputs interpretable as probabilities, differentiable
#    - Cons: Computationally expensive
#    - Use case: Output layer for multi-class classification
"""
            },
            {
                "title": "Implementing Custom Activation Functions in PyTorch",
                "description": "How would you implement custom activation functions in PyTorch?",
                "code": """
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Using nn.Module (for activation functions with parameters)
class PReLU(nn.Module):
    def __init__(self, alpha=0.01):
        super(PReLU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)

# Method 2: Using Functional API (for simpler functions)
def swish(x):
    return x * torch.sigmoid(x)

# Method 3: Using lambda functions
mish = lambda x: x * torch.tanh(F.softplus(x))

# Using in a neural network
class CustomNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.prelu = PReLU()  # Method 1
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.prelu(self.fc1(x))
        x = swish(self.fc2(x))  # Method 2
        x = self.fc3(x)
        x = mish(x)  # Method 3
        return x
"""
            }
        ],
        "resources": [
            {"title": "Understanding Activation Functions in Neural Networks", "url": "https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6"},
            {"title": "Activation Functions: Comparison of Trends in Practice and Research", "url": "https://arxiv.org/abs/1811.03378"},
            {"title": "Deep Learning Book - Chapter 6.3: Hidden Units", "url": "https://www.deeplearningbook.org/contents/mlp.html"}
        ],
        "related_topics": [
            "Neural Networks", "Backpropagation", "Vanishing Gradient Problem", "Weight Initialization", "Batch Normalization"
        ]
    } 