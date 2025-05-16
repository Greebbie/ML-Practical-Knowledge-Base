def get_content():
    return {
        "section": [
            {
                "title": "LoRA Fundamentals",
                "description": """
                <p>LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that reduces the number of trainable parameters by using low-rank decomposition.</p>
                <p>Key advantages:</p>
                <ul>
                    <li>Significantly reduces memory requirements</li>
                    <li>Enables multiple task-specific adapters</li>
                    <li>Maintains model performance</li>
                </ul>
                """,
                "formula": "$$h = Wx + \Delta Wx = Wx + (BA)x$$",
                "img": "img/lora_diagram.png",
                "caption": "Diagram illustrating LoRA's low-rank adaptation in neural networks."
            },
            {
                "title": "Implementation Details",
                "description": """
                <p>LoRA works by adding small trainable rank decomposition matrices to existing weights in the model.</p>
                <p>Implementation steps:</p>
                <ul>
                    <li>Identify target layers for adaptation</li>
                    <li>Create low-rank matrices</li>
                    <li>Apply during forward pass</li>
                </ul>
                """,
                "formula": "$$\Delta W = BA, \text{ where } B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, r \ll \min(d,k)$$",
                "example": "For a 768×768 weight matrix, LoRA might use two 768×8 matrices, reducing parameters from 589,824 to 12,288."
            }
        ],
        "implementation": """
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # Original forward pass
        original_output = self.original_layer(x)
        
        # LoRA forward pass
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        
        return original_output + lora_output

class LoRAModel(nn.Module):
    def __init__(self, base_model, target_modules, rank=4):
        super().__init__()
        self.base_model = base_model
        
        # Add LoRA layers to target modules
        for name, module in self.base_model.named_modules():
            if name in target_modules:
                # Replace original layer with LoRA layer
                setattr(self.base_model, name, LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=rank
                ))
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)
        """,
        "interview_examples": [
            {
                "title": "Implementing LoRA for Transformer Models",
                "description": "A practical example of implementing LoRA for fine-tuning transformer models.",
                "code": """
def apply_lora_to_transformer(model, rank=4):
    # Identify target layers
    target_modules = [
        'attention.query',
        'attention.key',
        'attention.value',
        'attention.output'
    ]
    
    # Create LoRA model
    lora_model = LoRAModel(model, target_modules, rank=rank)
    
    # Freeze base model parameters
    for param in lora_model.base_model.parameters():
        param.requires_grad = False
    
    # Only train LoRA parameters
    for name, param in lora_model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    
    return lora_model
                """
            }
        ],
        "resources": [
            {
                "title": "LoRA Paper",
                "url": "https://arxiv.org/abs/2106.09685"
            },
            {
                "title": "PEFT Library Documentation",
                "url": "https://huggingface.co/docs/peft/index"
            }
        ],
        "related_topics": [
            "Fine-tuning Techniques",
            "Parameter-Efficient Learning",
            "Model Adaptation",
            "Transfer Learning"
        ]
    } 