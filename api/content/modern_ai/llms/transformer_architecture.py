def get_content():
    return {
        "overview": """
        <p>The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need," revolutionized natural language processing by replacing recurrent networks with self-attention mechanisms.</p>
        <p>Key components:</p>
        <ul>
            <li>Encoder-Decoder structure</li>
            <li>Multi-head self-attention</li>
            <li>Positional encoding</li>
            <li>Feed-forward neural networks</li>
            <li>Residual connections and layer normalization</li>
        </ul>
        """,
        "core_concepts": [
            {
                "title": "Positional Encoding",
                "content": """
                <p>Unlike RNNs, Transformers process all tokens simultaneously, losing sequential information. Positional encodings inject position information into the embeddings.</p>
                <p>The standard positional encoding uses sine and cosine functions of different frequencies:</p>
                """,
                "formula": "PE_{(pos,2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right) \\\\ PE_{(pos,2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right)"
            },
            {
                "title": "Encoder-Decoder Structure",
                "content": """
                <p>The Transformer consists of an encoder that processes the input sequence and a decoder that generates the output sequence.</p>
                <p>Each encoder layer has:</p>
                <ul>
                    <li>Multi-head self-attention mechanism</li>
                    <li>Position-wise feed-forward network</li>
                    <li>Residual connections and layer normalization</li>
                </ul>
                <p>Each decoder layer has:</p>
                <ul>
                    <li>Masked multi-head self-attention</li>
                    <li>Multi-head cross-attention over encoder output</li>
                    <li>Position-wise feed-forward network</li>
                    <li>Residual connections and layer normalization</li>
                </ul>
                """
            },
            {
                "title": "Self-Attention Mechanism",
                "content": """
                <p>Self-attention allows the model to weigh the importance of different tokens in a sequence when processing each token. It's the core innovation of the Transformer architecture.</p>
                <p>Key components:</p>
                <ul>
                    <li><strong>Query (Q)</strong>: What we're looking for</li>
                    <li><strong>Key (K)</strong>: What we're matching against</li>
                    <li><strong>Value (V)</strong>: The actual content we want to extract</li>
                </ul>
                <p>The attention mechanism computes a weighted sum of values, where the weights are determined by the compatibility between the query and keys.</p>
                """,
                "formula": "\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V"
            },
            {
                "title": "Multi-Head Attention",
                "content": """
                <p>Multi-head attention allows the model to jointly attend to information from different positions and representation subspaces.</p>
                <p>Instead of performing a single attention function, the model projects queries, keys, and values into different subspaces and performs attention in parallel.</p>
                <p>This allows the model to capture different types of relationships between words:</p>
                <ul>
                    <li>Syntactic relationships</li>
                    <li>Semantic relationships</li>
                    <li>Contextual dependencies</li>
                </ul>
                <p>The outputs from different attention heads are concatenated and linearly transformed to produce the final result.</p>
                """,
                "formula": "\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\text{head}_2, ..., \\text{head}_h)W^O"
            },
            {
                "title": "Layer Normalization",
                "content": """
                <p>Layer normalization is applied after each sub-layer in the Transformer (after attention and after feed-forward network).</p>
                <p>It helps stabilize training by normalizing the inputs across the features:</p>
                <ul>
                    <li>Computes the mean and variance across the feature dimension</li>
                    <li>Normalizes each example independently</li>
                    <li>Applies learned scaling and shifting parameters</li>
                </ul>
                """,
                "formula": "\\text{LayerNorm}(x) = \\gamma \\odot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta \\\\ \\text{where } \\mu = \\frac{1}{n}\\sum_{i=1}^{n}x_i \\text{ and } \\sigma^2 = \\frac{1}{n}\\sum_{i=1}^{n}(x_i - \\mu)^2"
            },
            {
                "title": "Feed-Forward Networks",
                "content": """
                <p>Each Transformer layer contains a position-wise feed-forward neural network that is applied to each position in the sequence independently.</p>
                <p>This network consists of two linear transformations with a ReLU activation in between:</p>
                <ul>
                    <li>First linear layer expands the dimension</li>
                    <li>ReLU activation introduces non-linearity</li>
                    <li>Second linear layer projects back to the model dimension</li>
                </ul>
                """,
                "formula": "\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2"
            },
            {
                "title": "Residual Connections",
                "content": """
                <p>Residual connections (or skip connections) are used around each sub-layer in the Transformer.</p>
                <p>These connections help with:</p>
                <ul>
                    <li>Gradient flow during backpropagation</li>
                    <li>Training of deeper networks</li>
                    <li>Combining information from different layers</li>
                </ul>
                <p>The output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.</p>
                """,
                "formula": "\\text{Output} = \\text{LayerNorm}(x + \\text{Sublayer}(x))"
            }
        ],
        "implementation_code": """
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward block
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        # Add positional encoding to input embeddings
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))
        self.layers = nn.ModuleList(encoder_layers)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src shape: (batch_size, seq_len)
        src = self.embedding(src) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        
        for layer in self.layers:
            src = layer(src, src_mask, src_key_padding_mask)
            
        src = self.norm(src)
        return src.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
""",
        "interview_examples": [
            {
                "title": "Explain the advantages of Transformers over RNNs",
                "description": "What makes Transformers more effective than traditional RNN architectures for many NLP tasks?",
                "code": """
# Advantages of Transformers over RNNs:
# 1. Parallelization: Process all input tokens simultaneously, allowing faster training
# 2. Long-range dependencies: Self-attention directly connects all positions, avoiding the vanishing gradient problem
# 3. Constant path length: Information between any two positions travels through a constant number of operations
# 4. Interpretability: Attention weights provide insights into which input tokens are important for each output token
"""
            },
            {
                "title": "Implement a simple Transformer encoder",
                "description": "How would you implement a basic Transformer encoder with multi-head attention?",
                "code": """
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
    def forward(self, src, src_mask=None):
        # src shape: (seq_len, batch_size)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output
"""
            }
        ],
        "resources": [
            {"title": "Attention Is All You Need (original paper)", "url": "https://arxiv.org/abs/1706.03762"},
            {"title": "The Illustrated Transformer", "url": "http://jalammar.github.io/illustrated-transformer/"},
            {"title": "The Annotated Transformer", "url": "http://nlp.seas.harvard.edu/2018/04/03/attention.html"}
        ],
        "related_topics": [
            "Attention Mechanisms", "Self-Attention", "Large Language Models", "BERT", "GPT"
        ]
    } 