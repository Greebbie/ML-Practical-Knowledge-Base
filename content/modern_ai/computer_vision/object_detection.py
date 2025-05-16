def get_content():
    return {
        "section": [
            {
                "title": "Object Detection Fundamentals",
                "description": """
                <p>Object detection is a computer vision task that involves identifying and localizing objects within an image or video frame.</p>
                <p>Key components:</p>
                <ul>
                    <li>Bounding Box Prediction</li>
                    <li>Object Classification</li>
                    <li>Feature Extraction</li>
                    <li>Non-Maximum Suppression</li>
                </ul>
                """,
                "formula": "$$\\text{IoU} = \\frac{\\text{Area of Overlap}}{\\text{Area of Union}}$$",
                "img": "img/object_detection_example.png",
                "caption": "Example of object detection with bounding boxes and class labels."
            },
            {
                "title": "YOLO Architecture",
                "description": """
                <p>YOLO (You Only Look Once) is a popular real-time object detection system that processes images in a single forward pass.</p>
                <p>Key features:</p>
                <ul>
                    <li>Single-stage detection</li>
                    <li>Real-time performance</li>
                    <li>Grid-based prediction</li>
                    <li>Multi-scale detection</li>
                </ul>
                """,
                "formula": "$$\\text{Confidence} = P(\\text{Object}) \\times \\text{IoU}(\\text{pred}, \\text{truth})$$"
            }
        ],
        "implementation": """
import torch
import torch.nn as nn
import torchvision.models as models

class YOLOLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.grid_size = 0  # grid size will be set during forward pass

    def forward(self, x, targets=None):
        batch_size = x.size(0)
        grid_size = x.size(2)
        
        # Reshape predictions
        prediction = x.view(batch_size, self.num_anchors,
                          self.num_classes + 5, grid_size, grid_size)
        prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
        
        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        
        # Calculate offsets for each grid
        grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).float()
        grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).float()
        
        # Calculate anchor w, h
        anchor_w = self.anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = self.anchors[:, 1:2].view((1, self.num_anchors, 1, 1))
        
        # Add offset and scale with anchors
        pred_boxes = torch.zeros_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = torch.exp(w) * anchor_w
        pred_boxes[..., 3] = torch.exp(h) * anchor_h
        
        return pred_boxes, conf, pred_cls

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv3, self).__init__()
        # Load pretrained Darknet-53
        darknet = models.darknet53(pretrained=True)
        self.features = darknet.features
        
        # Detection layers
        self.yolo_layers = nn.ModuleList([
            YOLOLayer(anchors=[(10,13), (16,30), (33,23)], num_classes=num_classes),
            YOLOLayer(anchors=[(30,61), (62,45), (59,119)], num_classes=num_classes),
            YOLOLayer(anchors=[(116,90), (156,198), (373,326)], num_classes=num_classes)
        ])
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Get detections from each YOLO layer
        detections = []
        for yolo_layer in self.yolo_layers:
            detections.append(yolo_layer(features))
            
        return detections
        """,
        "interview_examples": [
            {
                "title": "Implementing Non-Maximum Suppression",
                "description": "A common interview question about implementing NMS for object detection.",
                "code": """
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    # boxes: (N, 4) where N is the number of boxes
    # scores: (N,) confidence scores for each box
    
    # Sort boxes by confidence score
    indices = torch.argsort(scores, descending=True)
    keep = []
    
    while indices.numel() > 0:
        # Get the box with highest confidence
        current = indices[0]
        keep.append(current)
        
        if indices.numel() == 1:
            break
            
        # Calculate IoU with remaining boxes
        remaining = indices[1:]
        ious = box_iou(boxes[current].unsqueeze(0), boxes[remaining])
        
        # Remove boxes with IoU > threshold
        mask = ious <= iou_threshold
        indices = remaining[mask]
    
    return torch.tensor(keep)

def box_iou(box1, box2):
    # box1: (N, 4), box2: (M, 4)
    # Returns: (N, M) IoU values
    
    # Get coordinates
    b1_x1, b1_y1 = box1[:, 0], box1[:, 1]
    b1_x2, b1_y2 = box1[:, 0] + box1[:, 2], box1[:, 1] + box1[:, 3]
    b2_x1, b2_y1 = box2[:, 0], box2[:, 1]
    b2_x2, b2_y2 = box2[:, 0] + box2[:, 2], box2[:, 1] + box2[:, 3]
    
    # Calculate intersection
    inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    
    # Calculate areas
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    # Calculate IoU
    union_area = b1_area.unsqueeze(1) + b2_area - inter_area
    iou = inter_area / union_area
    
    return iou
                """
            }
        ],
        "resources": [
            {
                "title": "YOLOv3 Paper",
                "url": "https://arxiv.org/abs/1804.02767"
            },
            {
                "title": "Object Detection with Deep Learning",
                "url": "https://www.tensorflow.org/tutorials/images/object_detection"
            }
        ],
        "related_topics": [
            "Computer Vision",
            "Deep Learning",
            "Image Processing",
            "Neural Networks"
        ]
    } 