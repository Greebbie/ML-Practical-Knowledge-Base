def get_content():
    return {
        "section": [
            {
                "title": "Attention Mechanisms: Core Concepts",
                "description": """
                <p>Attention mechanisms allow neural networks to focus on specific parts of input sequences when generating outputs. They are fundamental to modern NLP models, especially transformers.</p>
                <p>Key concepts:</p>
                <ul>
                    <li>Query, Key, Value (QKV) mechanism</li>
                    <li>Scaled dot-product attention</li>
                    <li>Multi-head attention</li>
                    <li>Self-attention vs. cross-attention</li>
                </ul>
                """,
                "formula": "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$",
                "img": "img/attention_diagram.png",
                "caption": "Diagram of self-attention mechanism in transformers."
            },
            {
                "title": "Self-Attention",
                "description": """
                <p>Self-attention allows a model to associate different positions of a single sequence. Each position attends to all positions in the sequence, capturing intra-sequence dependencies.</p>
                <p>The process:</p>
                <ol>
                    <li>Generate query (Q), key (K), and value (V) matrices for each input element</li>
                    <li>Calculate attention scores between each Q-K pair</li>
                    <li>Scale scores by dimension factor and apply softmax</li>
                    <li>Weight value vectors by attention scores</li>
                </ol>
                """,
                "formula": "$$\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) = \\frac{\\exp(\\frac{q_i \\cdot k_j}{\\sqrt{d_k}})}{\\sum_{l=1}^n \\exp(\\frac{q_i \\cdot k_l}{\\sqrt{d_k}})}$$"
            },
            {
                "title": "Multi-Head Attention",
                "description": """
                <p>Multi-head attention extends single-head attention by splitting the embedding dimension into multiple heads, allowing the model to attend to information from different representation subspaces.</p>
                """,
                "formula": "$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\ldots, \\text{head}_h)W^O$$\n$$\\text{where}\\; \\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$"
            }
        ],
        "implementation": """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)
        
        # Linear projections for Q, K, V, and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        
        # Reshape back to batch_size x seq_len x d_model
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        out = self.output_linear(out)
        
        return out, attention
""",
        "interview_examples": [
            {
                "title": "Explain the intuition behind scaled dot-product attention",
                "description": "Why is scaling by the square root of the embedding dimension important in attention mechanisms?",
                "code": """
# Scaling by sqrt(d_k) prevents the dot products from growing too large in magnitude, 
# which would push the softmax function into regions where gradients are extremely small.
# As the dimension of the embeddings (d_k) increases, dot products tend to have larger
# magnitudes, making the softmax distribution more peaked and reducing gradient flow.
"""
            },
            {
                "title": "Implement a simple self-attention layer",
                "description": "How would you implement a basic self-attention mechanism in PyTorch?",
                "code": """
def self_attention(query, key, value, mask=None):
    # query, key, value shape: (batch_size, seq_len, d_model)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights
"""
            }
        ],
        "resources": [
            {"title": "Attention Is All You Need (paper)", "url": "https://arxiv.org/abs/1706.03762"},
            {"title": "The Illustrated Transformer", "url": "http://jalammar.github.io/illustrated-transformer/"},
            {"title": "Visualizing Attention in Transformers", "url": "https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1"}
        ],
        "related_topics": [
            "Transformer Architecture", "Self-Attention", "Language Models", "Sequence-to-Sequence Models"
        ]
    } 