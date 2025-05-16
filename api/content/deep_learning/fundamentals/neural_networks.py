def get_content():
    return {
        "section": [
            {
                "title": "Neural Networks Fundamentals",
                "description": """
                <p>Neural networks are computational models inspired by the human brain, consisting of interconnected nodes (neurons) organized in layers. They are the foundation of deep learning.</p>
                <p>Key components:</p>
                <ul>
                    <li>Input Layer</li>
                    <li>Hidden Layers</li>
                    <li>Output Layer</li>
                    <li>Weights and Biases</li>
                    <li>Activation Functions</li>
                </ul>
                """,
                "formula": "$$y = f(\mathbf{W}x + b)$$",
                "img": "img/neural_network_diagram.png",
                "caption": "Diagram of a simple feedforward neural network."
            },
            {
                "title": "Feedforward and Backpropagation",
                "description": """
                <p>Feedforward neural networks pass data from input to output without cycles. Backpropagation is used to train the network by minimizing the loss function.</p>
                <p>Key steps:</p>
                <ul>
                    <li>Forward Pass</li>
                    <li>Loss Calculation</li>
                    <li>Backward Pass (Backpropagation)</li>
                    <li>Parameter Update</li>
                </ul>
                """,
                "formula": "$$L = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$"
            }
        ],
        "implementation": """
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    def relu(self, x):
        return np.maximum(0, x)
    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
""",
        "interview_examples": [
            {
                "title": "Explain the Universal Approximation Theorem",
                "description": "What does the Universal Approximation Theorem state about neural networks?",
                "code": """
# The Universal Approximation Theorem states that a feedforward neural network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of R^n, given appropriate activation functions.
"""
            }
        ],
        "resources": [
            {"title": "Neural Networks and Deep Learning (Book)", "url": "http://neuralnetworksanddeeplearning.com/"},
            {"title": "CS231n: Convolutional Neural Networks for Visual Recognition", "url": "http://cs231n.stanford.edu/"}
        ],
        "related_topics": [
            "Deep Learning", "Backpropagation", "Activation Functions", "Loss Functions"
        ]
    } 