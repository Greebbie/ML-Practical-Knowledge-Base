import os
import importlib.util
from pathlib import Path

def load_topic_content(topic_name):
    """
    Dynamically load topic content from the appropriate module.
    Topics are organized in a directory structure matching the main categories.
    """
    # Convert topic name to path format (e.g., 'transformer_architecture' -> 'deep_learning/architectures/transformer')
    topic_path = _get_topic_path(topic_name)
    
    # Try to load the specific topic module
    try:
        module_path = Path(__file__).parent / f"{topic_path}.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(topic_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.get_content()
    except Exception as e:
        print(f"Error loading topic {topic_name}: {e}")
    
    # Return default content if topic not found
    return _get_default_content()

def _get_topic_path(topic_name):
    """
    Convert topic name to its corresponding path in the content directory.
    This maps the topic structure defined in the main app to the file system.
    """
    # Mapping of topic names to their category paths
    topic_mapping = {
        # Deep Learning
        'neural_networks': 'deep_learning/fundamentals/neural_networks',
        'backpropagation': 'deep_learning/fundamentals/backpropagation',
        
        # Modern AI
        'transformer_architecture': 'modern_ai/llms/transformer_architecture',
        'lora_implementation': 'modern_ai/llms/lora',
        'attention_mechanisms': 'modern_ai/llms/attention',
        
        # Add more mappings as needed
    }
    
    return topic_mapping.get(topic_name, topic_name)

def _get_default_content():
    """Return default content structure for topics that don't have specific content yet."""
    return {
        "section": [
            {
                "title": "Core concept 1",
                "description": "This is a placeholder for the core concept.",
                "example": "This is an example of the concept."
            }
        ],
        "implementation": "Placeholder implementation",
        "interview_examples": [
            {
                "title": "Example Interview Question",
                "description": "This is a placeholder for an interview question.",
                "code": "def example():\n    pass"
            }
        ],
        "resources": [
            {
                "title": "Example Resource",
                "url": "#"
            }
        ],
        "related_topics": ["Related Topic 1", "Related Topic 2"]
    } 