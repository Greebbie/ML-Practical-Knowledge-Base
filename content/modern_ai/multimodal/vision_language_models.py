def get_content():
    return {
        "section": [
            {
                "title": "Vision-Language Models Fundamentals",
                "description": """
                <p>Vision-Language Models (VLMs) are AI systems that can understand and generate content across both visual and textual modalities.</p>
                <p>Key components:</p>
                <ul>
                    <li>Image Encoder</li>
                    <li>Text Encoder</li>
                    <li>Cross-Modal Fusion</li>
                    <li>Joint Training</li>
                </ul>
                """,
                "formula": "$$L = L_{\\text{vision}} + L_{\\text{text}} + L_{\\text{cross}}$$"
            },
            {
                "title": "CLIP Architecture",
                "description": """
                <p>CLIP (Contrastive Language-Image Pre-training) is a powerful VLM that learns visual concepts from natural language supervision.</p>
                <p>Key features:</p>
                <ul>
                    <li>Contrastive Learning</li>
                    <li>Zero-shot Transfer</li>
                    <li>Joint Embedding Space</li>
                    <li>Scalable Training</li>
                </ul>
                """,
                "formula": "$$\\text{sim}(I, T) = \\cos(f(I), g(T))$$",
                "img": "img/clip_architecture.png",
                "caption": "Diagram of the CLIP vision-language model architecture."
            }
        ],
        "implementation": """
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel

class CLIP(nn.Module):
    def __init__(self, vision_encoder='resnet50', text_encoder='bert-base-uncased'):
        super().__init__()
        
        # Initialize vision encoder
        if vision_encoder == 'resnet50':
            self.vision_encoder = models.resnet50(pretrained=True)
            self.vision_encoder.fc = nn.Linear(2048, 512)
        else:
            raise ValueError(f"Unsupported vision encoder: {vision_encoder}")
        
        # Initialize text encoder
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.text_projection = nn.Linear(768, 512)
        
        # Temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        # Extract image features
        features = self.vision_encoder(image)
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def encode_text(self, text):
        # Extract text features
        features = self.text_encoder(text)[0]
        features = self.text_projection(features[:, 0, :])
        features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def forward(self, image, text):
        # Get image and text features
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # Compute similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text

class VisionLanguageFusion(nn.Module):
    def __init__(self, vision_dim=512, text_dim=512, hidden_dim=1024):
        super().__init__()
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
    def forward(self, vision_features, text_features):
        # Concatenate features
        combined = torch.cat([vision_features, text_features], dim=-1)
        
        # Apply fusion
        fused = self.fusion(combined)
        
        # Apply self-attention
        fused = fused.unsqueeze(0)  # Add sequence dimension
        fused, _ = self.attention(fused, fused, fused)
        fused = fused.squeeze(0)  # Remove sequence dimension
        
        return fused

class VLMForImageCaptioning(nn.Module):
    def __init__(self, vision_encoder='resnet50', text_encoder='gpt2'):
        super().__init__()
        
        # Initialize encoders
        self.vision_encoder = models.resnet50(pretrained=True)
        self.vision_encoder.fc = nn.Linear(2048, 512)
        
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        self.text_decoder = AutoModel.from_pretrained(text_encoder)
        
        # Fusion module
        self.fusion = VisionLanguageFusion()
        
        # Output projection
        self.output = nn.Linear(1024, self.text_encoder.config.vocab_size)
        
    def forward(self, image, text_input_ids, text_attention_mask):
        # Encode image
        vision_features = self.vision_encoder(image)
        
        # Encode text
        text_features = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )[0]
        
        # Fuse features
        fused = self.fusion(vision_features, text_features)
        
        # Generate caption
        outputs = self.text_decoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            encoder_hidden_states=fused
        )
        
        # Project to vocabulary
        logits = self.output(outputs[0])
        
        return logits
        """,
        "interview_examples": [
            {
                "title": "Implementing Contrastive Learning for CLIP",
                "description": "A common interview question about implementing the contrastive learning objective for CLIP.",
                "code": """
def contrastive_loss(logits_per_image, logits_per_text):
    # Get batch size
    batch_size = logits_per_image.shape[0]
    
    # Create labels
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    # Compute loss
    loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2

def train_clip_step(model, image, text, optimizer):
    # Forward pass
    logits_per_image, logits_per_text = model(image, text)
    
    # Compute loss
    loss = contrastive_loss(logits_per_image, logits_per_text)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
                """
            }
        ],
        "resources": [
            {
                "title": "CLIP Paper",
                "url": "https://arxiv.org/abs/2103.00020"
            },
            {
                "title": "Vision-Language Models Survey",
                "url": "https://arxiv.org/abs/2202.09073"
            }
        ],
        "related_topics": [
            "Computer Vision",
            "Natural Language Processing",
            "Multimodal Learning",
            "Contrastive Learning"
        ]
    } 