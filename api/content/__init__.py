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

__all__ = ['load_content', 'load_topic_content'] 