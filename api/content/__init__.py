import sys
import os
import importlib.spec

# Try to directly import functions from content.py
try:
    # Get the absolute path to content.py
    content_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'content.py')
    
    if os.path.exists(content_path):
        # Load the content.py module directly using importlib
        spec = importlib.util.spec_from_file_location("content_module", content_path)
        content_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(content_module)
        
        # Copy the functions
        load_content = content_module.load_content
        load_topic_content = content_module.load_topic_content
    else:
        # Fallback definitions if content.py doesn't exist
        def load_content(topic_name):
            return [{"title": "Content Not Available", "description": f"<p>Content for {topic_name} could not be loaded.</p>", "subsections": []}]
            
        def load_topic_content(topic_name):
            return f"Content for {topic_name} is not available."
except Exception as e:
    # Fallback definitions if import fails
    def load_content(topic_name):
        return [{"title": "Error Loading Content", "description": f"<p>Error loading content: {str(e)}</p>", "subsections": []}]
        
    def load_topic_content(topic_name):
        return f"Error loading content: {str(e)}"

# Define the most essential functions directly in __init__.py
# This makes sure they're available when imported from api.content

def load_content(topic_name):
    """
    Load content for a specific topic.
    Returns a structured format compatible with the template.
    """
    # Simplified version that returns placeholder content
    topic_title = topic_name.replace('_', ' ').title()
    
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
                    "math": "f(x) = \\sum_{i=1}^{n} w_i x_i + b"
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
                    "code": """import numpy as np

def algorithm(data, params):
    # Initialize
    model = initialize_model(params)
    
    # Train
    for epoch in range(params['epochs']):
        output = model.forward(data)
        loss = calculate_loss(output, data.target)
        model.backward(loss)
        
    return model"""
                }
            ]
        }
    ]

def load_topic_content(topic_name):
    """
    A simple placeholder function that returns content for a given topic.
    """
    # Create a basic placeholder
    return {
        'section': [
            {
                'title': f'{topic_name.replace("_", " ").title()} Overview',
                'description': f'<p>This is an overview of {topic_name.replace("_", " ").title()}.</p>'
            }
        ],
        'resources': [
            {
                'title': f'Official Documentation for {topic_name.replace("_", " ").title()}',
                'url': f'https://docs.example.com/{topic_name}'
            },
            {
                'title': f'Tutorial on {topic_name.replace("_", " ").title()}',
                'url': f'https://tutorials.example.com/{topic_name}'
            }
        ],
        'related_topics': ['Deep Learning', 'Machine Learning', 'Neural Networks']
    }

__all__ = ['load_content', 'load_topic_content'] 