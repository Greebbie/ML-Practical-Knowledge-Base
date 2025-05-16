import os
import sys
import json
import importlib.util
import random

def load_content(topic_name):
    """
    Load content for a specific topic.
    Returns a structured format compatible with the new template.
    """
    # Standard structured content format
    try:
        # First try to load from content module
        content_module_name = topic_name.lower().replace(' ', '_')
        
        # Try different import approaches (local dev vs. Vercel)
        try:
            # Try to load from old topic content approach
            old_content = load_topic_content(topic_name)
            if isinstance(old_content, dict) and 'section' in old_content:
                return convert_old_content_format(old_content, topic_name)
        except Exception as e:
            print(f"Error loading from old format: {e}")
            
        try:
            # First method - direct import from content directory
            module_path = os.path.join('api', 'content', 'topics.py')
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location("topics", module_path)
                topics_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(topics_module)
                
                if hasattr(topics_module, content_module_name):
                    return getattr(topics_module, content_module_name)
        except Exception as e:
            print(f"Error loading from topics module: {e}")
        
        # If no specific content is found, generate a structured placeholder
        return generate_placeholder_content(topic_name)
        
    except Exception as e:
        print(f"Error loading content for {topic_name}: {e}")
        return None

def convert_old_content_format(old_content, topic_name):
    """Convert old content format to new structure"""
    new_content = []
    
    # Add overview section
    new_content.append({
        "title": "Introduction",
        "description": old_content.get('section', [])[0].get('description', f"<p>Overview of {topic_name}</p>") if old_content.get('section') else f"<p>Overview of {topic_name}</p>",
        "subsections": []
    })
    
    # Add core concepts sections
    sections = []
    for section in old_content.get('section', []):
        subsection = {
            "title": section.get('title', 'Concept'),
            "content": section.get('description', '')
        }
        
        if 'formula' in section:
            subsection["math"] = section['formula'].replace('$', '')
            
        if 'implementation' in section:
            subsection["code"] = section['implementation']
            
        sections.append(subsection)
    
    if sections:
        new_content.append({
            "title": "Concepts",
            "description": "<p>Key concepts and theoretical foundations.</p>",
            "subsections": sections
        })
    
    # Add implementation section if available
    implementations = []
    for section in old_content.get('section', []):
        if 'implementation' in section:
            implementations.append({
                "title": "Code Implementation",
                "content": f"<p>Implementation for {section.get('title', 'this concept')}:</p>",
                "code": section['implementation']
            })
    
    if implementations:
        new_content.append({
            "title": "Implementation Approaches",
            "description": "<p>Practical implementations of the concepts.</p>",
            "subsections": implementations
        })
    
    return new_content

def generate_placeholder_content(topic_name):
    """
    Generate placeholder content for a topic that doesn't have specific content yet.
    """
    topic_title = topic_name.replace('_', ' ').title()
    
    # Get examples based on topic name to make more specific
    math_example = get_topic_specific_math(topic_name)
    code_example = get_topic_specific_code(topic_name)
    
    # Default sections for any topic
    return [
        {
            "title": "Introduction",
            "description": f"<p>{topic_title} is an important topic in machine learning and artificial intelligence. This section provides an overview of key concepts and applications.</p>",
            "subsections": [
                {
                    "title": "Overview",
                    "content": f"<p>{topic_title} refers to a set of techniques and methods that are widely used in the field. These methods provide solutions to complex problems by leveraging mathematical and computational principles.</p>"
                },
                {
                    "title": "Key Concepts",
                    "content": "<p>The following are fundamental concepts that form the basis of this topic:</p><ul><li>Theoretical foundations</li><li>Algorithmic implementations</li><li>Performance considerations</li><li>Real-world applications</li></ul>"
                }
            ]
        },
        {
            "title": "Theoretical Background",
            "description": "<p>Understanding the mathematical and theoretical foundations is essential for mastering this topic.</p>",
            "subsections": [
                {
                    "title": "Mathematical Foundation",
                    "content": "<p>The mathematical principles that underlie this topic include:</p><ul><li>Linear algebra concepts</li><li>Probabilistic frameworks</li><li>Optimization techniques</li></ul>",
                    "math": math_example
                }
            ]
        },
        {
            "title": "Implementation Approaches",
            "description": "<p>There are various ways to implement and apply these concepts in practice.</p>",
            "subsections": [
                {
                    "title": "Algorithmic Approach",
                    "content": "<p>The typical algorithm follows these steps:</p><ol><li>Data preparation and normalization</li><li>Model initialization</li><li>Iterative optimization</li><li>Evaluation and refinement</li></ol>"
                },
                {
                    "title": "Code Example",
                    "content": "<p>A simple implementation might look like this:</p>",
                    "code": code_example
                }
            ]
        }
    ]

def get_topic_specific_math(topic_name):
    """Return math formula relevant to the topic"""
    topic_key = topic_name.lower()
    
    math_examples = {
        "neural_networks": "y = \\sigma(\\sum_{i=1}^{n} w_i x_i + b)",
        "backpropagation": "\\frac{\\partial L}{\\partial w_{ij}} = \\frac{\\partial L}{\\partial y_j} \\frac{\\partial y_j}{\\partial w_{ij}}",
        "transformer": "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
        "attention": "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
        "linear_regression": "h_\\theta(x) = \\theta_0 + \\theta_1 x_1 + \\ldots + \\theta_n x_n",
        "logistic_regression": "P(y=1|x;\\theta) = \\frac{1}{1 + e^{-\\theta^T x}}",
        "svm": "\\min_{w, b} \\frac{1}{2}||w||^2 \\text{ subject to } y_i(w^T x_i + b) \\geq 1",
        "decision_trees": "\\text{Gini}(D) = 1 - \\sum_{i=1}^{n} p_i^2",
        "random_forests": "\\text{OOB Error} = \\frac{1}{n} \\sum_{i=1}^{n} I(\\hat{y}_i \\neq y_i)",
        "reinforcement_learning": "Q(s, a) \\leftarrow (1 - \\alpha) \\cdot Q(s, a) + \\alpha \\cdot (r + \\gamma \\cdot \\max_{a'} Q(s', a'))",
        "clustering": "J = \\sum_{j=1}^{k} \\sum_{i=1}^{n} ||x_i^{(j)} - c_j||^2",
        "pca": "\\Sigma = \\frac{1}{n} \\sum_{i=1}^{n} (x_i - \\mu)(x_i - \\mu)^T"
    }
    
    # Check for partial matches
    for key, formula in math_examples.items():
        if key in topic_key:
            return formula
    
    # Default formula
    return "f(x) = \\sum_{i=1}^{n} w_i x_i + b"

def get_topic_specific_code(topic_name):
    """Return code example relevant to the topic"""
    topic_key = topic_name.lower()
    
    code_examples = {
        "neural_networks": """import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        
    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2""",
        
        "transformer": """import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        # Get batch size
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out""",
        
        "linear_regression": """import numpy as np

def linear_regression(X, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    theta = np.zeros(X.shape[1])
    
    for i in range(iterations):
        # Compute predictions
        predictions = np.dot(X, theta)
        
        # Compute error
        error = predictions - y
        
        # Update parameters
        gradient = (1/m) * np.dot(X.T, error)
        theta = theta - learning_rate * gradient
        
    return theta"""
    }
    
    # Check for partial matches
    for key, code in code_examples.items():
        if key in topic_key:
            return code
    
    # Default code example
    return """import numpy as np

def algorithm(data, params):
    # Initialize
    model = initialize_model(params)
    
    # Train
    for epoch in range(params['epochs']):
        output = model.forward(data)
        loss = calculate_loss(output, data.target)
        model.backward(loss)
        
    return model"""

def load_topic_content(topic_name):
    """
    A simple placeholder function that returns content for a given topic.
    In a real application, this would load from a database or files.
    """
    # Create a basic template for neural networks as an example
    if topic_name == "neural_networks":
        return {
            'section': [
                {
                    'title': 'Neural Networks Fundamentals',
                    'description': '''
                    <p>Neural networks are computational models inspired by the human brain, consisting of
                    interconnected nodes (neurons) organized in layers. They are the foundation of deep learning.</p>
                    <p>Key components:</p>
                    <ul>
                        <li>Input Layer</li>
                        <li>Hidden Layers</li>
                        <li>Output Layer</li>
                        <li>Weights and Biases</li>
                        <li>Activation Functions</li>
                    </ul>
                    ''',
                    'formula': '$y = f(\\mathbf{W}x + b)$',
                    'img': 'img/neural_network_diagram.png',
                    'caption': 'Diagram of a simple feedforward neural network.'
                },
                {
                    'title': 'Feedforward and Backpropagation',
                    'description': '''
                    <p>Feedforward neural networks pass data from input to output without cycles. 
                    Backpropagation is used to train the network by minimizing the loss function.</p>
                    <p>Key steps:</p>
                    <ul>
                        <li>Forward Pass</li>
                        <li>Loss Computation</li>
                        <li>Backward Pass (Backpropagation)</li>
                        <li>Parameter Update</li>
                    </ul>
                    ''',
                    'formula': '$\\delta L = \\nabla{f}(N)\\cdot \\nabla{y}_{j}^2$$',
                    'implementation': '''
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
                    '''
                }
            ],
            'interview_examples': [
                {
                    'title': 'Implement Self-Attention from Scratch',
                    'description': 'A common interview question about implementing the attention mechanism without using PyTorch\'s built-in functions.',
                    'code': '''
def self_attention_scratch(Q, K, V, mask=None):
    # Q, K, V shape: (batch_size, seq_len, d_k)
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
                    '''
                },
                {
                    'title': 'Explain the Universal Approximation Theorem',
                    'description': 'What does the Universal Approximation Theorem state about neural networks?',
                    'code': '''
# The Universal Approximation Theorem states that a feedforward neural network 
# with a single hidden layer containing a finite number of neurons can approximate 
# any continuous function on compact subsets of R^n, given appropriate activation functions.
                    '''
                }
            ],
            'resources': [
                {
                    'title': 'Neural Networks and Deep Learning (Book)',
                    'url': 'http://neuralnetworksanddeeplearning.com/'
                },
                {
                    'title': 'CS231n: Convolutional Neural Networks for Visual Recognition',
                    'url': 'http://cs231n.stanford.edu/'
                }
            ],
            'related_topics': ['Deep Learning', 'Backpropagation', 'Activation Functions', 'Loss Functions']
        }
    elif topic_name == "transformer_architecture":
        return {
            'section': [
                {
                    'title': 'Transformer Architecture Overview',
                    'description': '''
                    <p>The Transformer is a model architecture that relies entirely on attention mechanisms
                    to draw global dependencies between input and output. It was introduced in the paper
                    "Attention Is All You Need" and has become the foundation for most modern NLP systems.</p>
                    <p>Key components:</p>
                    <ul>
                        <li>Self-Attention Mechanism</li>
                        <li>Multi-Head Attention</li>
                        <li>Positional Encoding</li>
                        <li>Feed-Forward Networks</li>
                        <li>Layer Normalization</li>
                    </ul>
                    ''',
                    'formula': '$Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$',
                },
                {
                    'title': 'Self-Attention Mechanism',
                    'description': '''
                    <p>The self-attention mechanism is the core innovation of the Transformer architecture. It allows the model to
                    weigh the importance of different words in a sequence when processing each word.</p>
                    <p>Key components:</p>
                    <ul>
                        <li>Query (Q): What we're looking for</li>
                        <li>Key (K): What we're matching against</li>
                        <li>Value (V): The actual content we want to extract</li>
                    </ul>
                    ''',
                    'formula': '$Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$',
                    'implementation': '''
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        
        # Reshape and concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        return self.out(out)
                    '''
                },
                {
                    'title': 'Multi-Head Attention',
                    'description': '''
                    <p>Multi-head attention allows the model to attend to information from different positions
                    using different representation subspaces. This enriches the model's ability to capture
                    relationships between words.</p>
                    ''',
                    'formula': '$MultiHead(Q, K, V) = Concat(head_1, head_2, ..., head_h)W^O$',
                },
                {
                    'title': 'Positional Encoding',
                    'description': '''
                    <p>Since the Transformer architecture doesn't have any recurrence or convolution,
                    positional encoding is added to the input embeddings to provide information about
                    the position of tokens in the sequence.</p>
                    ''',
                    'formula': '$PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}})$<br>$PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d_{model}})$',
                }
            ],
            'interview_examples': [
                {
                    'title': 'Implementing Self-Attention from Scratch',
                    'description': 'A common interview question about implementing the attention mechanism without using PyTorch\'s built-in functions.',
                    'code': '''
def self_attention_scratch(Q, K, V, mask=None):
    # Q, K, V shape: (batch_size, seq_len, d_k)
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.size(-1))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Apply attention weights to values
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights
                    '''
                }
            ],
            'resources': [
                {
                    'title': 'Original Transformer Paper',
                    'url': 'https://arxiv.org/abs/1706.03762'
                },
                {
                    'title': 'Attention Is All You Need - Explained',
                    'url': 'https://jalammar.github.io/illustrated-transformer/'
                },
                {
                    'title': 'The Annotated Transformer',
                    'url': 'https://nlp.seas.harvard.edu/2018/04/03/attention.html'
                }
            ],
            'related_topics': ['Attention Mechanisms', 'Multi-head Attention', 'Positional Encoding', 'Encoder-Decoder Architecture']
        }
    else:
        return f"Content for {topic_name} is not yet available." 