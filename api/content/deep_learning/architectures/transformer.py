def get_content():
    """Return the content for the Transformer Architecture topic."""
    return {
        "overview": """
        <p>The Transformer is a model architecture that relies entirely on attention mechanisms
        to draw global dependencies between input and output. It was introduced in the paper
        "Attention Is All You Need" and has become the foundation for most modern NLP systems.</p>
        
        <p>Unlike previous sequence models that relied on recurrence or convolution, Transformers
        use a mechanism called self-attention to process input sequences in parallel, leading to
        significant improvements in both training efficiency and performance on various tasks.</p>
        """,
        "core_concepts": [
            {
                "title": "Self-Attention Mechanism",
                "content": """
                <p>The self-attention mechanism is the core innovation of the Transformer architecture. It allows the model to
                weigh the importance of different words in a sequence when processing each word.</p>
                
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
                <p>Multi-head attention allows the model to jointly attend to information from different positions and representation subspaces. Instead of performing a single attention function, the model projects the queries, keys, and values into different subspaces and performs attention in parallel.</p>
                
                <p>This allows the model to capture different types of relationships between words:</p>
                <ul>
                    <li>Syntactic relationships</li>
                    <li>Semantic relationships</li>
                    <li>Contextual dependencies</li>
                </ul>
                
                <p>The outputs from different attention heads are concatenated and linearly transformed to produce the final result.</p>
                """,
                "formula": "\\text{MultiHead}(Q,K,V) = \\text{Concat}(\\text{head}_1, \\text{head}_2, ..., \\text{head}_h)W^O \\\\\\text{where } \\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)"
            },
            {
                "title": "Positional Encoding",
                "content": """
                <p>Since Transformers don't have inherent understanding of sequence order, positional encodings are added
                to give the model information about the relative or absolute position of tokens in the sequence.</p>
                
                <p>Key features:</p>
                <ul>
                    <li>Uses sine and cosine functions of different frequencies</li>
                    <li>Allows the model to generalize to sequences of different lengths</li>
                    <li>Enables the model to learn relative positions</li>
                </ul>
                
                <p>Example: In the sequence 'I love AI', the positional encoding helps distinguish between 'I love AI' and 'AI love I'.</p>
                """,
                "formula": "PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right) \\\\PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d_{\\text{model}}}}\\right)"
            },
            {
                "title": "Feed-Forward Networks",
                "content": """
                <p>Each Transformer layer contains a position-wise feed-forward neural network that is applied to each position in the sequence independently. This network consists of two linear transformations with a ReLU activation in between.</p>
                
                <p>The feed-forward network allows the model to:</p>
                <ul>
                    <li>Process features from attention mechanisms</li>
                    <li>Introduce non-linearity into the model</li>
                    <li>Transform the representations individually for each position</li>
                </ul>
                """,
                "formula": "\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2"
            },
            {
                "title": "Layer Normalization",
                "content": """
                <p>Layer normalization is applied after each sub-layer in the Transformer (after attention and after feed-forward network). It helps stabilize training by normalizing the inputs across the features.</p>
                
                <p>Layer normalization:</p>
                <ul>
                    <li>Computes the mean and variance across the feature dimension</li>
                    <li>Normalizes each example independently</li>
                    <li>Applies learned scaling and shifting parameters</li>
                </ul>
                
                <p>This contrasts with batch normalization, which normalizes across the batch dimension.</p>
                """,
                "formula": "\\text{LayerNorm}(x) = \\gamma \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta"
            },
            {
                "title": "Residual Connections",
                "content": """
                <p>Residual connections (or skip connections) are used around each sub-layer in the Transformer. They help with the flow of gradients through the network, enabling training of deeper models.</p>
                
                <p>The output of each sub-layer is:</p>
                <ul>
                    <li>LayerNorm(x + Sublayer(x))</li>
                </ul>
                
                <p>Where Sublayer(x) is the function implemented by the sub-layer itself (attention or feed-forward network).</p>
                """,
                "formula": "\\text{Output} = \\text{LayerNorm}(x + \\text{Sublayer}(x))"
            }
        ],
        "implementation_code": """
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size must be divisible by heads"
        
        # Linear projections
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, head_dim)
        # out shape: (N, query_len, heads, head_dim) -> (N, query_len, embed_size)
        
        out = self.fc_out(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but should be saved)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input embeddings
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        # Add skip connection, run through normalization and dropout
        x = self.norm1(attention + query)
        x = self.dropout(x)
        
        # Feed forward
        forward = self.feed_forward(x)
        
        # Again add skip connection, run through normalization and dropout
        out = self.norm2(forward + x)
        out = self.dropout(out)
        return out
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
            },
            {
                "title": "Training a Transformer Model",
                "description": "How to set up training for a Transformer model, focusing on key considerations.",
                "code": """
# Define model, loss function, and optimizer
model = TransformerModel(vocab_size=10000, embed_size=512, num_layers=6, heads=8, 
                         device="cuda", forward_expansion=4, dropout=0.1, max_len=100)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Get batch data
        src = batch.src.to(device)  # (batch_size, src_len)
        trg = batch.trg.to(device)  # (batch_size, trg_len)
        
        # Create masks
        src_mask = model.make_src_mask(src)  # (batch_size, 1, 1, src_len)
        trg_mask = model.make_trg_mask(trg)  # (batch_size, 1, trg_len, trg_len)
        
        # Forward pass
        output = model(src, trg, src_mask, trg_mask)  # (batch_size, trg_len, vocab_size)
        
        # Calculate loss (ignore padding)
        output = output.reshape(-1, output.shape[2])  # (batch_size * trg_len, vocab_size)
        trg = trg[:, 1:].reshape(-1)  # Exclude padding
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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