from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
import importlib.util
import sys

# Dynamic import of content module - works in both local and Vercel environments
try:
    # First try direct import (works locally)
    from content import load_topic_content
except ImportError:
    try:
        # Then try package-style import (may work on Vercel)
        from api.content import load_topic_content
    except ImportError:
        # Finally, try loading the module directly from the file path
        content_path = os.path.join(os.path.dirname(__file__), "content.py")
        spec = importlib.util.spec_from_file_location("content", content_path)
        content = importlib.util.module_from_spec(spec)
        sys.modules["content"] = content
        spec.loader.exec_module(content)
        load_topic_content = content.load_topic_content

# Determine if running on Vercel
IS_VERCEL = os.environ.get('VERCEL') == '1'

# Simple Flask app configuration
# We're letting Vercel handle static files through the routes in vercel.json
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
    try:
        return render_template('base.html', topics=topics)
    except Exception as e:
        # Return a simple response instead of crashing with 500
        import traceback
        error_message = f"Error rendering homepage: {str(e)}\n{traceback.format_exc()}"
        print(error_message)  # This will appear in Vercel logs
        return f"""
        <html>
        <head><title>ML Knowledge Base - Error</title></head>
        <body>
            <h1>Something went wrong</h1>
            <p>The application encountered an error. Please try again later.</p>
            <pre>{error_message}</pre>
        </body>
        </html>
        """

@app.route('/topic/<topic_name>')
def topic_page(topic_name):
    try:
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
    except Exception as e:
        # Return a simple response instead of crashing with 500
        import traceback
        error_message = f"Error rendering topic page for {topic_name}: {str(e)}\n{traceback.format_exc()}"
        print(error_message)  # This will appear in Vercel logs
        return f"""
        <html>
        <head><title>ML Knowledge Base - Error</title></head>
        <body>
            <h1>Something went wrong</h1>
            <p>The application encountered an error while loading topic: {topic_name}</p>
            <p>Please try again later or <a href="/">return to homepage</a>.</p>
            <pre>{error_message}</pre>
        </body>
        </html>
        """

def generate_questions(topic):
    """
    Generate domain-specific questions based on topic
    Can be implemented using GPT-3/LLMs or custom templates
    """
    pass

# No app.run() here for Vercel; local_dev.py handles local execution.
# The 'app' variable is picked up by Vercel automatically from api/index.py 