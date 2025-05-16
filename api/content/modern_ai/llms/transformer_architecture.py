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
                "title": "Implementing Multi-Head Attention from Scratch",
                "description": "Extending the self-attention mechanism to implement multi-head attention, another common interview question.",
                "code": """
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return x.transpose(1, 2)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections
        q = self.W_q(q)  # (batch_size, seq_len, d_model)
        k = self.W_k(k)  # (batch_size, seq_len, d_model)
        v = self.W_v(v)  # (batch_size, seq_len, d_model)
        
        # Split heads
        q = self.split_heads(q)  # (batch_size, num_heads, seq_len, d_k)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len, d_k)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len, d_k)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Add extra dimensions to mask for broadcasting
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, d_k)
        
        # Transpose and reshape back
        context = context.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, d_k)
        context = context.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output, attention_weights
                """
            },
            {
                "title": "Creating a Causal Mask for Decoder Self-Attention",
                "description": "How to implement a causal mask to prevent attending to future positions, crucial for autoregressive generation.",
                "code": """
def generate_causal_mask(seq_len, device="cpu"):
    # Create a square mask with ones on and below the diagonal, zeros elsewhere
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    # Invert the mask: 1s are positions to keep, 0s are positions to mask
    mask = ~mask
    
    # Move to device and add batch dimension for broadcasting
    return mask.to(device).unsqueeze(0)  # (1, seq_len, seq_len)

# Example usage:
seq_len = 5
causal_mask = generate_causal_mask(seq_len)
print(causal_mask[0])

# Output:
# tensor([[ True, False, False, False, False],
#         [ True,  True, False, False, False],
#         [ True,  True,  True, False, False],
#         [ True,  True,  True,  True, False],
#         [ True,  True,  True,  True,  True]])
                """
            },
            {
                "title": "Visualizing Attention Patterns",
                "description": "How to extract and visualize attention patterns, useful for model interpretability and debugging.",
                "code": """
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(model, tokenizer, sentence, layer_idx=0, head_idx=0):
    # Tokenize input
    tokens = tokenizer.tokenize(sentence)
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    
    # Forward pass with output_attentions=True
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
    
    # Extract attention weights from specified layer and head
    # Shape: (batch_size, num_heads, seq_len, seq_len)
    attention = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()
    
    # Plot attention matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, 
                xticklabels=tokens, 
                yticklabels=tokens, 
                cmap="viridis", 
                annot=True)
    plt.title(f"Attention Matrix (Layer {layer_idx}, Head {head_idx})")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.show()

# Example usage (with a hypothetical model and tokenizer):
# visualize_attention(model, tokenizer, "The transformer architecture is powerful.")
                """
            },
            {
                "title": "Implementing Flash Attention for Memory Efficiency",
                "description": "Optimizing attention computation for long sequences using the Flash Attention approach.",
                "code": """
def flash_attention(Q, K, V, mask=None, chunk_size=4096):
    \"\"\"
    Memory-efficient attention implementation for long sequences.
    Processes attention in chunks to reduce memory usage.
    
    Args:
        Q, K, V: Query, Key, Value tensors (batch_size, seq_len, d_k)
        mask: Optional mask tensor (batch_size, seq_len, seq_len)
        chunk_size: Size of chunks to process at once
    \"\"\"
    batch_size, seq_len, d_k = Q.size()
    
    # Initialize output and attention weights
    output = torch.zeros_like(Q)
    attention_weights = torch.zeros(batch_size, seq_len, seq_len, device=Q.device)
    
    # Process in chunks to save memory
    for i in range(0, seq_len, chunk_size):
        # Get current chunk of queries
        q_chunk = Q[:, i:i+chunk_size, :]
        chunk_len = q_chunk.size(1)
        
        for j in range(0, seq_len, chunk_size):
            # Get current chunk of keys and values
            k_chunk = K[:, j:j+chunk_size, :]
            v_chunk = V[:, j:j+chunk_size, :]
            
            # Compute chunk of attention scores
            scores = torch.bmm(q_chunk, k_chunk.transpose(1, 2)) / math.sqrt(d_k)
            
            # Apply mask if provided
            if mask is not None:
                chunk_mask = mask[:, i:i+chunk_size, j:j+chunk_size]
                scores = scores.masked_fill(chunk_mask == 0, -1e9)
            
            # Store attention weights for visualization
            attention_weights[:, i:i+chunk_size, j:j+chunk_size] = torch.softmax(scores, dim=-1)
            
            # For output computation, we need to normalize across all chunks
            if j == 0:
                # First chunk - initialize max values and weights
                max_scores = scores.max(dim=-1, keepdim=True)[0]
                exp_scores = torch.exp(scores - max_scores)
                exp_weights_sum = exp_scores.sum(dim=-1, keepdim=True)
                weighted_values = torch.bmm(exp_scores, v_chunk)
            else:
                # Update max values and weights with new chunk
                new_max_scores = torch.max(max_scores, scores.max(dim=-1, keepdim=True)[0])
                exp_scores_old = exp_scores * torch.exp(max_scores - new_max_scores)
                exp_scores_new = torch.exp(scores - new_max_scores)
                
                exp_weights_sum = exp_weights_sum * torch.exp(max_scores - new_max_scores) + exp_scores_new.sum(dim=-1, keepdim=True)
                weighted_values = weighted_values * torch.exp(max_scores - new_max_scores) + torch.bmm(exp_scores_new, v_chunk)
                
                max_scores = new_max_scores
                exp_scores = exp_scores_new
                
        # Compute final output for current query chunk
        output[:, i:i+chunk_size, :] = weighted_values / exp_weights_sum
        
    return output, attention_weights
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