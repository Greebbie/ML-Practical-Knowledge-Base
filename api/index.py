from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
# Import from the local content.py file
from content import load_topic_content

# Determine if running on Vercel
IS_VERCEL = os.environ.get('VERCEL') == '1'

if IS_VERCEL:
    # On Vercel, Flask should not handle static files.
    # Vercel's routing rules (in vercel.json) will serve them from the 'public' directory.
    app = Flask(__name__)
else:
    # For local development, Flask needs to serve static files.
    # The 'public' directory is at the project root, so relative to 'api/index.py',
    # the static folder is '../public'. The URL path remains '/static'.
    app = Flask(__name__, static_url_path='/static', static_folder='../public/static')

# Content structure
topics = {
    'deep_learning': {
        'fundamentals': [
            'Neural Networks',
            'Backpropagation',
            'Activation Functions',
            'Loss Functions',
            'Optimization Algorithms'
        ],
        'architectures': [
            'CNNs',
            'RNNs',
            'Transformers',
            'GANs',
            'Autoencoders'
        ],
        'advanced': [
            'Transfer Learning',
            'Few-shot Learning',
            'Meta Learning',
            'Neural Architecture Search',
            'Model Compression'
        ]
    },
    'machine_learning': {
        'supervised': [
            'Linear Regression',
            'SVM',
            'Decision Trees',
            'Random Forests',
            'Gradient Boosting'
        ],
        'unsupervised': [
            'Clustering',
            'Dimensionality Reduction',
            'Anomaly Detection',
            'Association Rules',
            'Topic Modeling'
        ],
        'reinforcement': [
            'Q-Learning',
            'Policy Gradients',
            'Multi-agent Systems',
            'Deep RL',
            'Inverse RL'
        ]
    },
    'modern_ai': {
        'llms': [
            'Transformer Architecture',
            'Attention Mechanisms',
            'Fine-tuning Techniques',
            'LoRA Implementation',
            'Prompt Engineering'
        ],
        'computer_vision': [
            'Object Detection',
            'Image Segmentation',
            'Video Understanding',
            '3D Vision',
            'Neural Rendering'
        ],
        'multimodal': [
            'Vision-Language Models',
            'Audio-Visual Learning',
            'Cross-modal Retrieval',
            'Multimodal Fusion',
            'Zero-shot Learning'
        ]
    },
    'math_foundations': {
        'linear_algebra': [
            'Matrices',
            'Eigenvectors',
            'Vector Spaces',
            'Matrix Decomposition',
            'Tensor Operations'
        ],
        'calculus': [
            'Gradients',
            'Chain Rule',
            'Optimization',
            'Vector Calculus',
            'Numerical Methods'
        ],
        'probability': [
            'Distributions',
            'Bayesian Methods',
            'Information Theory',
            'Statistical Inference',
            'Stochastic Processes'
        ]
    }
}

@app.route('/')
def home():
    return render_template('base.html', topics=topics)

@app.route('/topic/<topic_name>')
def topic_page(topic_name):
    # Ensure content loading doesn't break if 'content.py' isn't fully implemented
    try:
        content_data = load_topic_content(topic_name)
    except Exception as e:
        print(f"Error loading topic content for {topic_name}: {e}")
        content_data = f"Content for {topic_name} is not yet available."
    
    # If content_data is a string (simple output), use it directly
    # If it's a dictionary (structured content), pass it to the template
    is_structured = isinstance(content_data, dict)
    
    summary = "This is a detailed explanation of the topic with practical examples and implementations."
    questions = [
        {
            "text": "What is the key concept behind this topic?",
            "difficulty": "easy",
            "hint": "Think about the fundamental principles"
        },
        {
            "text": "How would you implement this in a production environment?",
            "difficulty": "hard",
            "hint": "Consider scalability and efficiency"
        }
    ]
    return render_template('topic.html',
                         content=content_data,
                         is_structured=is_structured,
                         summary=summary,
                         questions=questions,
                         topic_name=topic_name)

def generate_questions(topic):
    """
    Generate domain-specific questions based on topic
    Can be implemented using GPT-3/LLMs or custom templates
    """
    pass

# No app.run() here for Vercel; local_dev.py handles local execution.
# The 'app' variable is picked up by Vercel automatically from api/index.py 