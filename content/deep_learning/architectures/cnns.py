def get_content():
    return {
        "section": [
            {
                "title": "Convolutional Neural Networks: Overview",
                "description": """
                <p>Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed to process grid-like data, particularly images. They leverage spatial locality through convolutional operations, reducing parameters compared to fully connected networks.</p>
                <p>Key components of CNNs:</p>
                <ul>
                    <li>Convolutional layers</li>
                    <li>Pooling layers</li>
                    <li>Activation functions</li>
                    <li>Fully connected layers</li>
                    <li>Batch normalization</li>
                </ul>
                """,
                "img": "img/cnn_architecture.png",
                "caption": "Typical CNN architecture showing convolutional, pooling, and fully connected layers."
            },
            {
                "title": "Convolutional Layers",
                "description": """
                <p>Convolutional layers apply a set of learnable filters (kernels) to input data. Each filter slides across the input, computing dot products to produce feature maps that highlight specific patterns.</p>
                <p>Key hyperparameters:</p>
                <ul>
                    <li><strong>Kernel Size</strong>: The spatial dimensions of the filter (e.g., 3×3, 5×5)</li>
                    <li><strong>Stride</strong>: The step size when sliding the filter</li>
                    <li><strong>Padding</strong>: Adding zeros around the input to preserve spatial dimensions</li>
                    <li><strong>Number of Filters</strong>: Determines the depth of the output volume</li>
                </ul>
                """,
                "formula": """
                $$\\text{Output Height} = \\frac{\\text{Input Height} - \\text{Filter Height} + 2 \\times \\text{Padding}}{\\text{Stride}} + 1$$
                
                $$\\text{Output Width} = \\frac{\\text{Input Width} - \\text{Filter Width} + 2 \\times \\text{Padding}}{\\text{Stride}} + 1$$
                
                $$\\text{Convolution Operation:} \\; (I * K)_{i,j} = \\sum_{m}\\sum_{n} I_{i+m, j+n} \\cdot K_{m,n}$$
                """
            },
            {
                "title": "Pooling Layers",
                "description": """
                <p>Pooling layers reduce the spatial dimensions of feature maps, decreasing computational complexity and introducing translational invariance. They summarize features in local neighborhoods.</p>
                <p>Common pooling operations:</p>
                <ul>
                    <li><strong>Max Pooling</strong>: Selects the maximum value from each region</li>
                    <li><strong>Average Pooling</strong>: Computes the average of values in each region</li>
                    <li><strong>Global Pooling</strong>: Pools over the entire feature map, often used before fully connected layers</li>
                </ul>
                """,
                "formula": """
                $$\\text{Max Pooling:} \\; \\text{Out}_{i,j} = \\max_{m,n \\in R_{i,j}} x_{m,n}$$
                
                $$\\text{Average Pooling:} \\; \\text{Out}_{i,j} = \\frac{1}{|R_{i,j}|} \\sum_{m,n \\in R_{i,j}} x_{m,n}$$
                """
            },
            {
                "title": "Classic CNN Architectures",
                "description": """
                <p>Several influential CNN architectures have advanced the field:</p>
                <ul>
                    <li><strong>LeNet-5 (1998)</strong>: Early CNN for digit recognition</li>
                    <li><strong>AlexNet (2012)</strong>: Deeper network that won ImageNet, using ReLU activations and dropout</li>
                    <li><strong>VGGNet (2014)</strong>: Emphasized simplicity with consistent 3×3 filters and increasing depth</li>
                    <li><strong>GoogLeNet/Inception (2014)</strong>: Introduced inception modules with multiple filter sizes</li>
                    <li><strong>ResNet (2015)</strong>: Added skip connections to train very deep networks</li>
                    <li><strong>DenseNet (2017)</strong>: Connected each layer to every other layer in a feed-forward fashion</li>
                </ul>
                """
            },
            {
                "title": "ResNet and Skip Connections",
                "description": """
                <p>ResNet (Residual Networks) introduced skip connections to address the vanishing gradient problem, allowing training of much deeper networks.</p>
                <p>The key insight is learning residual mappings by adding the input to the output of a block:</p>
                """,
                "formula": """
                $$F(x) + x \\text{ instead of } F(x)$$
                
                $$\\text{where } F(x) \\text{ represents a sequence of layers}$$
                """
            },
            {
                "title": "Applications of CNNs",
                "description": """
                <p>CNNs are widely used across computer vision tasks:</p>
                <ul>
                    <li><strong>Image Classification</strong>: Identifying objects in images</li>
                    <li><strong>Object Detection</strong>: Locating and classifying multiple objects (YOLO, SSD, Faster R-CNN)</li>
                    <li><strong>Semantic Segmentation</strong>: Labeling each pixel with a category (U-Net, FCN)</li>
                    <li><strong>Instance Segmentation</strong>: Detecting and segmenting individual objects (Mask R-CNN)</li>
                    <li><strong>Face Recognition</strong>: Identifying individuals from facial features</li>
                    <li><strong>Style Transfer</strong>: Transferring artistic styles between images</li>
                    <li><strong>Medical Image Analysis</strong>: Diagnosing conditions from medical scans</li>
                </ul>
                """
            },
            {
                "title": "Advanced Concepts in CNNs",
                "description": """
                <p>Modern CNN research explores several advanced concepts:</p>
                <ul>
                    <li><strong>Attention Mechanisms</strong>: Focusing on important regions (Squeeze-and-Excitation Networks)</li>
                    <li><strong>Dilated/Atrous Convolutions</strong>: Expanding receptive field without increasing parameters</li>
                    <li><strong>Depthwise Separable Convolutions</strong>: Factorizing standard convolution for efficiency (MobileNet)</li>
                    <li><strong>Group Convolutions</strong>: Dividing channels into groups to reduce computations (ResNeXt)</li>
                    <li><strong>Neural Architecture Search</strong>: Automatically discovering optimal CNN architectures</li>
                </ul>
                """,
                "formula": """
                $$\\text{Dilated Convolution:} \\; (I *_l K)_{i,j} = \\sum_{m}\\sum_{n} I_{i+l\\cdot m, j+l\\cdot n} \\cdot K_{m,n}$$
                
                $$\\text{where } l \\text{ is the dilation rate}$$
                """
            }
        ],
        "implementation": """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Define a simple CNN for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second convolutional block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# Define a ResNet block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

# Visualizing convolution operations
def visualize_convolution():
    # Create a sample image (grayscale)
    image = np.zeros((8, 8))
    image[2:6, 2:6] = 1  # White square in the middle
    
    # Define a vertical edge detection filter
    kernel_vertical = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    
    # Define a horizontal edge detection filter
    kernel_horizontal = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    
    # Perform convolution
    from scipy import signal
    output_vertical = signal.convolve2d(image, kernel_vertical, mode='valid')
    output_horizontal = signal.convolve2d(image, kernel_horizontal, mode='valid')
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(output_vertical, cmap='gray')
    plt.title('Vertical Edge Detection')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(output_horizontal, cmap='gray')
    plt.title('Horizontal Edge Detection')
    plt.colorbar()
    
    plt.tight_layout()

# Visualizing feature maps
def visualize_feature_maps():
    # Load a pretrained model
    model = torchvision.models.resnet18(pretrained=True)
    
    # Register hooks to get intermediate activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for the first few layers
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.layer1[0].conv1.register_forward_hook(get_activation('layer1.0.conv1'))
    model.layer1[0].conv2.register_forward_hook(get_activation('layer1.0.conv2'))
    
    # Load and preprocess an image
    input_image = torch.randn(1, 3, 224, 224)  # Random input for demonstration
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_image)
    
    # Visualize the activations
    plt.figure(figsize=(12, 8))
    
    # Original image
    plt.subplot(2, 2, 1)
    img = input_image[0].permute(1, 2, 0).numpy()
    img = (img - img.min()) / (img.max() - img.min())
    plt.imshow(img)
    plt.title('Input Image')
    
    # First layer activations
    plt.subplot(2, 2, 2)
    feature_map = activations['conv1'][0, 0].numpy()
    plt.imshow(feature_map, cmap='viridis')
    plt.title('Conv1 Feature Map (Channel 0)')
    
    # Later layer activations
    plt.subplot(2, 2, 3)
    feature_map = activations['layer1.0.conv1'][0, 0].numpy()
    plt.imshow(feature_map, cmap='viridis')
    plt.title('Layer1.0.Conv1 Feature Map (Channel 0)')
    
    plt.subplot(2, 2, 4)
    feature_map = activations['layer1.0.conv2'][0, 0].numpy()
    plt.imshow(feature_map, cmap='viridis')
    plt.title('Layer1.0.Conv2 Feature Map (Channel 0)')
    
    plt.tight_layout()

# Example of using a modern CNN architecture
def use_pretrained_model():
    # Load pretrained ResNet
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    
    # Prepare image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Example inference code (not executed)
    # image = Image.open('sample.jpg')
    # input_tensor = transform(image).unsqueeze(0)
    # with torch.no_grad():
    #     output = model(input_tensor)
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # top5_prob, top5_catid = torch.topk(probabilities, 5)
""",
        "interview_examples": [
            {
                "title": "Comparing CNN Architectures",
                "description": "How would you compare different CNN architectures and choose one for a specific vision task?",
                "code": """
# CNN Architecture Comparison:

# 1. VGG
#    - Architecture: Stacks of 3x3 convolutions followed by max pooling, very deep and uniform
#    - Pros: Simple design, strong feature extraction, great for transfer learning
#    - Cons: Computationally expensive, large model size, high memory usage
#    - Use case: When simplicity is valued and computational resources are not limited

# 2. ResNet
#    - Architecture: Introduces skip connections (residual blocks) to train very deep networks
#    - Pros: Can be extremely deep (50+ layers), solves vanishing gradient, high accuracy
#    - Cons: More complex than VGG, increasing depth gives diminishing returns
#    - Use case: General purpose, when high accuracy is needed, good default choice

# 3. MobileNet
#    - Architecture: Uses depthwise separable convolutions to reduce computations
#    - Pros: Fast inference, small model size, good for mobile/embedded devices
#    - Cons: Lower accuracy than larger models
#    - Use case: Edge devices, real-time applications with limited resources

# 4. EfficientNet
#    - Architecture: Uses compound scaling to balance depth, width, and resolution
#    - Pros: State-of-the-art accuracy with fewer parameters, efficient scaling
#    - Cons: Complex architecture, newer and less established
#    - Use case: When both accuracy and efficiency are important

# 5. Vision Transformer (ViT)
#    - Architecture: Adapts transformer architecture from NLP to vision
#    - Pros: Great for larger datasets, captures global dependencies
#    - Cons: Requires more data than CNNs, computationally intensive
#    - Use case: When you have large amounts of data and computational resources

def choose_architecture(task_requirements):
    if task_requirements["computational_constraints"] == "severe":
        if task_requirements["accuracy_requirements"] == "high":
            return "EfficientNet-B0 or MobileNetV3"
        else:
            return "MobileNetV2 or ShuffleNet"
            
    elif task_requirements["task"] == "object_detection":
        if task_requirements["real_time"]:
            return "SSD with MobileNet or YOLO"
        else:
            return "Faster R-CNN with ResNet backbone"
            
    elif task_requirements["task"] == "segmentation":
        if task_requirements["medical_imaging"]:
            return "U-Net"
        else:
            return "DeepLabV3+ with Xception backbone"
            
    elif task_requirements["data_amount"] == "limited":
        return "ResNet-50 with pretrained weights and fine-tuning"
        
    elif task_requirements["training_time"] == "limited":
        return "EfficientNet or ResNet-50"
        
    else:  # Default for general image classification
        return "ResNet-50 or EfficientNetB3"
"""
            },
            {
                "title": "Implementing a Custom CNN Architecture",
                "description": "Design and implement a custom CNN architecture for a specific task.",
                "code": """
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=3):
        super(CustomCNN, self).__init__()
        
        # First convolutional block with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block with batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block with batch normalization
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Feature pyramid to capture multi-scale features
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv4_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers with dropout
        fc_input_size = 64 * 3  # From the three parallel convolutions
        self.fc1 = nn.Linear(fc_input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Second block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Third block
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Multi-scale feature extraction
        x1 = F.relu(self.conv4_1(x))
        x2 = F.relu(self.conv4_2(x))
        x3 = F.relu(self.conv4_3(x))
        
        # Global pooling on each path
        x1 = self.global_pool(x1).view(-1, 64)
        x2 = self.global_pool(x2).view(-1, 64)
        x3 = self.global_pool(x3).view(-1, 64)
        
        # Concatenate features from different scales
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Custom implementation of a ResNet-like architecture with attention
class ResidualAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualAttentionBlock, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Squeeze-and-Excitation attention
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(out_channels, out_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 16, out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Residual path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Attention mechanism
        se = self.squeeze(out)
        se = se.view(se.size(0), -1)
        se = self.excitation(se)
        se = se.view(se.size(0), -1, 1, 1)
        out = out * se
        
        # Skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out

# Train and evaluate the model
def train_custom_cnn(train_loader, val_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomCNN(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        # Print statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Adjust learning rate
        scheduler.step(val_loss)
    
    return model
"""
            }
        ],
        "resources": [
            {"title": "CS231n: Convolutional Neural Networks for Visual Recognition", "url": "http://cs231n.stanford.edu/"},
            {"title": "Deep Learning for Computer Vision with Python", "url": "https://www.pyimagesearch.com/deep-learning-computer-vision-python-book/"},
            {"title": "A guide to convolution arithmetic for deep learning", "url": "https://arxiv.org/abs/1603.07285"}
        ],
        "related_topics": [
            "Computer Vision", "Image Classification", "Object Detection", "Semantic Segmentation", "Transfer Learning"
        ]
    } 