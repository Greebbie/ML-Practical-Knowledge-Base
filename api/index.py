from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from transformers import pipeline
import torch
import torch.nn as nn
import math
from content import load_topic_content

app = Flask(__name__)

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
    content = load_topic_content(topic_name)
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
                         content=content, 
                         summary=summary, 
                         questions=questions, 
                         topic_name=topic_name)

def generate_questions(topic):
    """
    Generate domain-specific questions based on topic
    Can be implemented using GPT-3/LLMs or custom templates
    """
    pass

# This is important for Vercel
app = app 