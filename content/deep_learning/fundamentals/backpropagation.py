def get_content():
    return {
        "section": [
            {
                "title": "Backpropagation Fundamentals",
                "description": """
                <p>Backpropagation is the fundamental algorithm for training neural networks, allowing them to learn from data by adjusting their weights.</p>
                <p>Key components:</p>
                <ul>
                    <li>Forward Pass</li>
                    <li>Loss Computation</li>
                    <li>Gradient Calculation</li>
                    <li>Weight Updates</li>
                </ul>
                """,
                "formula": "$$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial y} \\times \\frac{\\partial y}{\\partial w}$$",
                "example": "In a simple neural network, backpropagation calculates how each weight contributes to the final error."
            },
            {
                "title": "Chain Rule in Backpropagation",
                "description": """
                <p>The chain rule is essential for computing gradients in neural networks with multiple layers.</p>
                <p>Key concepts:</p>
                <ul>
                    <li>Local Gradients</li>
                    <li>Upstream Gradients</li>
                    <li>Gradient Flow</li>
                    <li>Computational Graph</li>
                </ul>
                """,
                "formula": "$$\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial y} \\times \\frac{\\partial y}{\\partial x}$$",
                "img": "img/chain_rule_graph.png",
                "caption": "Visualization of the chain rule in a computational graph."
            }
        ],
        "implementation": """
import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.input = None
        self.output = None
        
    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias
        return self.output
    
    def backward(self, grad_output, learning_rate):
        # Compute gradients
        grad_weights = np.dot(self.input.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Update weights and bias
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

class ReLU:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.activations = []
        
        # Create layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Don't add activation after last layer
                self.activations.append(ReLU())
    
    def forward(self, x):
        # Forward pass through layers
        for layer, activation in zip(self.layers[:-1], self.activations):
            x = layer.forward(x)
            x = activation.forward(x)
        
        # Last layer without activation
        x = self.layers[-1].forward(x)
        return x
    
    def backward(self, grad_output, learning_rate):
        # Backward pass through layers
        grad = grad_output
        
        # Backward through last layer
        grad = self.layers[-1].backward(grad, learning_rate)
        
        # Backward through remaining layers
        for layer, activation in zip(reversed(self.layers[:-1]), reversed(self.activations)):
            grad = activation.backward(grad)
            grad = layer.backward(grad, learning_rate)
        
        return grad

def compute_loss(y_pred, y_true):
    # Mean Squared Error
    return np.mean((y_pred - y_true) ** 2)

def compute_gradient(y_pred, y_true):
    # Gradient of MSE
    return 2 * (y_pred - y_true) / y_pred.shape[0]

def train_step(model, x, y, learning_rate):
    # Forward pass
    y_pred = model.forward(x)
    
    # Compute loss
    loss = compute_loss(y_pred, y)
    
    # Compute gradient
    grad = compute_gradient(y_pred, y)
    
    # Backward pass
    model.backward(grad, learning_rate)
    
    return loss
        """,
        "interview_examples": [
            {
                "title": "Implementing Backpropagation from Scratch",
                "description": "A common interview question about implementing backpropagation for a simple neural network.",
                "code": """
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize biases
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
    
    def forward(self, x):
        # Forward pass
        self.hidden = sigmoid(np.dot(x, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output
    
    def backward(self, x, y, learning_rate):
        # Compute gradients
        output_error = y - self.output
        output_delta = output_error * sigmoid_derivative(self.output)
        
        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)
        
        # Update weights and biases
        self.weights2 += learning_rate * np.dot(self.hidden.T, output_delta)
        self.bias2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        
        self.weights1 += learning_rate * np.dot(x.T, hidden_delta)
        self.bias1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)
    
    def train(self, x, y, learning_rate=0.1, epochs=1000):
        for _ in range(epochs):
            # Forward pass
            output = self.forward(x)
            
            # Backward pass
            self.backward(x, y, learning_rate)
            
            # Print loss every 100 epochs
            if _ % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {_}, Loss: {loss}")
                """
            }
        ],
        "resources": [
            {
                "title": "Backpropagation Paper",
                "url": "https://www.nature.com/articles/323533a0"
            },
            {
                "title": "Deep Learning Book - Backpropagation",
                "url": "https://www.deeplearningbook.org/contents/mlp.html"
            }
        ],
        "related_topics": [
            "Neural Networks",
            "Deep Learning",
            "Optimization",
            "Gradient Descent"
        ]
    } 