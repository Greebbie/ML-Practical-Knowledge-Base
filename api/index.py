import os
import sys
import random
import traceback
from flask import Flask, render_template, redirect, url_for, request, abort, send_from_directory, jsonify

# Simple import for Vercel - this is the only one that should be needed
from api.content import load_content, load_topic_content

app = Flask(__name__)

# Configure app to serve static files from the public directory
@app.route('/favicon.ico')
def favicon():
    return send_from_directory('public', 'favicon.ico')

@app.route('/style.css')
def css():
    return send_from_directory('public', 'style.css')

# Define the topic structure
topics = {
    "deep_learning": {
        "fundamentals": ["Neural Networks", "Backpropagation", "Activation Functions", "Loss Functions", "Optimization Algorithms"],
        "architectures": ["CNNs", "RNNs", "Transformers", "GANs", "Autoencoders"],
        "advanced": ["Transfer Learning", "Few-shot Learning", "Meta Learning", "Neural Architecture Search", "Model Compression"]
    },
    "machine_learning": {
        "supervised": ["Linear Regression", "SVM", "Decision Trees", "Random Forests", "Gradient Boosting"],
        "unsupervised": ["Clustering", "Dimensionality Reduction", "Anomaly Detection", "Association Rules", "Topic Modeling"],
        "reinforcement": ["Q-Learning", "Policy Gradients", "Multi-agent Systems", "Deep RL", "Inverse RL"]
    },
    "modern_ai": {
        "llms": ["Transformer Architecture", "Attention Mechanisms", "Fine-tuning Techniques", "LoRA Implementation", "Prompt Engineering"],
        "computer_vision": ["Object Detection", "Image Segmentation", "Video Understanding", "3D Vision", "Neural Rendering"],
        "multimodal": ["Vision-Language Models", "Audio-Visual Learning", "Cross-modal Retrieval", "Multimodal Fusion", "Zero-shot Learning"]
    },
    "math_foundations": {
        "linear_algebra": ["Matrices", "Vectors", "Eigenvalues", "SVD", "Vector Spaces"]
    }
}

# Define routes
@app.route('/')
def home():
    try:
        return render_template('index.html', topics=topics)
    except Exception as e:
        app.logger.error(f"Error rendering home page: {str(e)}")
        app.logger.error(traceback.format_exc())
        return f"Error rendering page: {str(e)}", 500

@app.route('/topic/<topic_name>')
def topic(topic_name):
    try:
        # Format title
        title = topic_name.replace('_', ' ').title()
        
        # Load specific content first directly from topics.py using the old content loader
        old_content = None
        try:
            old_content = load_topic_content(topic_name)
        except Exception as e:
            app.logger.error(f"Error loading old topic format: {str(e)}")
        
        # Check if we get the new format with core_concepts directly
        if old_content and isinstance(old_content, dict) and 'core_concepts' in old_content:
            # Use the new format
            core_concepts = old_content['core_concepts']
            overview = old_content.get('overview', '')
            implementation_code = old_content.get('implementation_code', '')
            interview_examples = old_content.get('interview_examples', [])
            resources = old_content.get('resources', [])
            related_topics = old_content.get('related_topics', [])
            
            # Generate questions related to the topic
            questions = generate_questions(topic_name, 3)
            
            return render_template('topic.html', 
                                  title=title,
                                  overview=overview,
                                  core_concepts=core_concepts,
                                  implementation_code=implementation_code,
                                  questions=questions,
                                  resources=resources,
                                  interview_examples=interview_examples,
                                  related_topics=related_topics)
        
        # Use the old fallback structure
        content = load_content(topic_name)
        if not content:
            return render_template('error.html', message=f"Topic '{topic_name}' not found"), 404
        
        # Generate questions related to the topic
        questions = generate_questions(topic_name, 3)
        
        # Try to get interview examples and related topics
        interview_examples = []
        related_topics = []
        resources = []
        
        # Try to load from old format if needed
        if isinstance(old_content, dict):
            interview_examples = old_content.get('interview_examples', [])
            related_topics = old_content.get('related_topics', [])
            
            # Use resources from old format if available
            if 'resources' in old_content:
                resources = [
                    {"title": item['title'], "url": item['url']} 
                    for item in old_content['resources']
                ]
        
        # If no resources were set yet, use example resources
        if not resources:
            resources = [
                {"title": f"Official {title} Documentation", "url": f"https://docs.example.com/{topic_name}"},
                {"title": f"Tutorial: Understanding {title}", "url": f"https://tutorials.example.com/{topic_name}"},
                {"title": f"Research Paper on {title}", "url": f"https://papers.example.com/{topic_name}.pdf"}
            ]
        
        return render_template('topic.html', 
                              title=title,
                              content=content,
                              questions=questions,
                              resources=resources,
                              interview_examples=interview_examples,
                              related_topics=related_topics)
    
    except Exception as e:
        app.logger.error(f"Error rendering topic {topic_name}: {str(e)}")
        app.logger.error(traceback.format_exc())
        return f"Error loading topic: {str(e)}", 500

@app.route('/resources')
def resources():
    try:
        return render_template('resources.html')
    except Exception as e:
        return f"Error loading resources: {str(e)}", 500

@app.route('/code')
def code_examples():
    try:
        return render_template('code_examples.html')
    except Exception as e:
        return f"Error loading code examples: {str(e)}", 500

# Helper functions
def generate_questions(topic, count=3):
    """Generate domain-specific questions based on the topic."""
    questions = []
    
    # Dictionary of questions for different domains
    domain_questions = {
        "neural_networks": [
            {"question": "Explain how backpropagation works in a neural network", "difficulty": "Medium", 
             "math": "\\frac{\\partial L}{\\partial w_{ij}} = \\frac{\\partial L}{\\partial y_j} \\frac{\\partial y_j}{\\partial w_{ij}}",
             "hint": "Think about the chain rule from calculus"},
            {"question": "How does vanishing gradient problem affect deep networks?", "difficulty": "Hard",
             "hint": "Consider what happens to gradients in very deep networks with certain activation functions"},
            {"question": "Implement a simple feedforward neural network using NumPy", "difficulty": "Hard", 
             "hint": "Break it down into initialization, forward pass, and backward pass",
             "code": "import numpy as np\n\ndef sigmoid(x):\n    return 1 / (1 + np.exp(-x))\n\ndef sigmoid_derivative(x):\n    return x * (1 - x)\n\nclass NeuralNetwork:\n    def __init__(self, x, y):\n        self.input = x\n        self.weights1 = np.random.rand(self.input.shape[1], 4)\n        self.weights2 = np.random.rand(4, 1)\n        self.y = y\n        self.output = np.zeros(y.shape)\n\n    def feedforward(self):\n        self.layer1 = sigmoid(np.dot(self.input, self.weights1))\n        self.output = sigmoid(np.dot(self.layer1, self.weights2))\n\n    def backprop(self):\n        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))\n        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), \n                                                self.weights2.T) * sigmoid_derivative(self.layer1))\n\n        self.weights1 += d_weights1\n        self.weights2 += d_weights2"},
        ],
        "transformers": [
            {"question": "Explain the multi-head attention mechanism in transformers", "difficulty": "Hard", 
             "math": "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
             "hint": "Think about why multiple attention heads are better than just one"},
            {"question": "How does positional encoding work in transformers?", "difficulty": "Medium",
             "hint": "Consider how transformers need position information since they have no recurrence or convolution"},
            {"question": "Why is layer normalization important in transformer architectures?", "difficulty": "Medium",
             "hint": "Think about training stability and convergence"}
        ],
        "linear_regression": [
            {"question": "Derive the normal equation for linear regression", "difficulty": "Medium", 
             "math": "\\hat{\\beta} = (X^TX)^{-1}X^Ty",
             "hint": "Think about minimizing the sum of squared errors"},
            {"question": "Explain the difference between L1 and L2 regularization", "difficulty": "Easy",
             "hint": "Consider their effects on the model parameters and feature selection"},
            {"question": "Implement linear regression with gradient descent", "difficulty": "Medium", 
             "hint": "Remember to compute the gradient of the cost function",
             "code": "import numpy as np\n\ndef gradient_descent(X, y, learning_rate=0.01, iterations=1000):\n    m = len(y)\n    theta = np.zeros(X.shape[1])\n    cost_history = []\n    \n    for i in range(iterations):\n        prediction = np.dot(X, theta)\n        error = prediction - y\n        cost = (1/(2*m)) * np.sum(error**2)\n        cost_history.append(cost)\n        \n        # Update theta\n        theta = theta - (learning_rate/m) * np.dot(X.T, error)\n    \n    return theta, cost_history"}
        ],
    }
    
    # Map the URL topic_name to the domain questions
    domain_mapping = {
        "neural_networks": "neural_networks",
        "backpropagation": "neural_networks",
        "activation_functions": "neural_networks",
        "transformers": "transformers",
        "attention_mechanisms": "transformers",
        "transformer_architecture": "transformers",
        "linear_regression": "linear_regression",
    }
    
    # Convert topic with underscores and handle case
    topic_key = topic.lower().replace(' ', '_')
    
    if topic_key in domain_mapping:
        question_pool = domain_questions[domain_mapping[topic_key]]
        # Return all questions if count is greater than available questions
        return random.sample(question_pool, min(count, len(question_pool)))
    
    # Generic questions if specific domain is not found
    generic_questions = [
        {"question": f"Explain the core concepts of {topic.replace('_', ' ').title()}", "difficulty": "Easy",
         "hint": "Think about the fundamental principles"},
        {"question": f"What are the practical applications of {topic.replace('_', ' ').title()}?", "difficulty": "Medium",
         "hint": "Consider both academic and industry use cases"},
        {"question": f"How would you implement this in a production environment?", "difficulty": "Hard",
         "hint": "Consider scalability and efficiency"}
    ]
    
    return random.sample(generic_questions, min(count, len(generic_questions)))

if __name__ == "__main__":
    app.run(debug=True) 