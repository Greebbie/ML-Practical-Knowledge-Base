def get_content():
    """Return the content for the Transformer Architecture topic."""
    return {
        "section": [
            {
                "title": "Self-Attention Mechanism",
                "description": """
                <p>The self-attention mechanism is the core innovation of the Transformer architecture. It allows the model to weigh the importance of different words in a sequence when processing each word.</p>
                <p>Key components:</p>
                <ul>
                    <li>Query (Q): What we're looking for</li>
                    <li>Key (K): What we're matching against</li>
                    <li>Value (V): The actual content we want to extract</li>
                </ul>
                """,
                "formula": "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$",
                "img": "img/self_attention_diagram.png",
                "caption": "Diagram of the self-attention mechanism in a Transformer."
            },
            {
                "title": "Multi-Head Attention",
                "description": """
                <p>Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.</p>
                <p>Benefits:</p>
                <ul>
                    <li>Allows the model to focus on different aspects of the input</li>
                    <li>Enables parallel processing of attention</li>
                    <li>Increases the model's capacity to learn different types of relationships</li>
                </ul>
                """,
                "formula": "$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O$$",
                "example": "In a sentence about a sports game, one head might focus on player actions while another focuses on the game context."
            },
            {
                "title": "Positional Encoding",
                "description": """
                <p>Since Transformers don't have inherent understanding of sequence order, positional encodings are added to give the model information about the relative or absolute position of tokens in the sequence.</p>
                <p>Key features:</p>
                <ul>
                    <li>Uses sine and cosine functions of different frequencies</li>
                    <li>Allows the model to generalize to sequences of different lengths</li>
                    <li>Enables the model to learn relative positions</li>
                </ul>
                """,
                "formula": "$$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right)$$",
                "example": "In the sequence 'I love AI', the positional encoding helps distinguish between 'I love AI' and 'AI love I'."
            }
        ],
        "implementation": """
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        
        # Reshape and apply final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output)
        """,
        "interview_examples": [
            {
                "title": "Implementing Self-Attention from Scratch",
                "description": "A common interview question about implementing the attention mechanism without using PyTorch's built-in functions.",
                "code": """
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
                """
            }
        ],
        "resources": [
            {
                "title": "Original Transformer Paper",
                "url": "https://arxiv.org/abs/1706.03762"
            },
            {
                "title": "Attention is All You Need - Explained",
                "url": "https://jalammar.github.io/illustrated-transformer/"
            },
            {
                "title": "The Annotated Transformer",
                "url": "http://nlp.seas.harvard.edu/2018/04/03/attention.html"
            }
        ],
        "related_topics": [
            "Attention Mechanisms",
            "Multi-head Attention",
            "Positional Encoding",
            "Encoder-Decoder Architecture"
        ]
    } 