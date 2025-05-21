import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Tuple

class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        
    def _one_hot_encode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert label indices to one-hot encoded format."""
        return F.one_hot(tensor.long(), self.num_classes).permute(0, 4, 1, 2, 3).float()
    
    def dice_coefficient(self, y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
        """
        Calculate Dice coefficient for each class.
        Args:
            y_pred: Predicted probabilities of shape (B, C, H, W, D)
            y_true: Ground truth labels of shape (B, H, W, D)
            smooth: Smoothing factor to avoid division by zero
        Returns:
            Dice coefficient for each class
        """
        y_true = self._one_hot_encode(y_true)
        
        intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3, 4))
        union = torch.sum(y_pred, dim=(0, 2, 3, 4)) + torch.sum(y_true, dim=(0, 2, 3, 4))
        dice = (2. * intersection + smooth) / (union + smooth)
        
        return dice
    
    def iou_score(self, y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-7) -> torch.Tensor:
        """
        Calculate IoU (Intersection over Union) score for each class.
        Args:
            y_pred: Predicted probabilities of shape (B, C, H, W, D)
            y_true: Ground truth labels of shape (B, H, W, D)
            smooth: Smoothing factor to avoid division by zero
        Returns:
            IoU score for each class
        """
        y_true = self._one_hot_encode(y_true)
        
        intersection = torch.sum(y_pred * y_true, dim=(0, 2, 3, 4))
        union = torch.sum(y_pred, dim=(0, 2, 3, 4)) + torch.sum(y_true, dim=(0, 2, 3, 4)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        return iou
    
    def accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        """
        Calculate pixel-wise accuracy.
        Args:
            y_pred: Predicted probabilities of shape (B, C, H, W, D)
            y_true: Ground truth labels of shape (B, H, W, D)
        Returns:
            Accuracy score
        """
        y_pred_labels = torch.argmax(y_pred, dim=1)
        correct = torch.sum(y_pred_labels == y_true)
        total = y_true.numel()
        return (correct / total).item()
    
    def class_wise_accuracy(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate class-wise accuracy.
        Args:
            y_pred: Predicted probabilities of shape (B, C, H, W, D)
            y_true: Ground truth labels of shape (B, H, W, D)
        Returns:
            Accuracy score for each class
        """
        y_true_one_hot = self._one_hot_encode(y_true)
        y_pred_labels = torch.argmax(y_pred, dim=1)
        y_pred_one_hot = self._one_hot_encode(y_pred_labels)
        
        correct_per_class = torch.sum(y_true_one_hot * y_pred_one_hot, dim=(0, 2, 3, 4))
        total_per_class = torch.sum(y_true_one_hot, dim=(0, 2, 3, 4))
        
        return correct_per_class / (total_per_class + 1e-7)
    
    def evaluate_batch(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        """
        Evaluate all metrics for a batch of predictions.
        Args:
            y_pred: Predicted probabilities of shape (B, C, H, W, D)
            y_true: Ground truth labels of shape (B, H, W, D)
        Returns:
            Dictionary containing all metrics
        """
        dice_scores = self.dice_coefficient(y_pred, y_true)
        iou_scores = self.iou_score(y_pred, y_true)
        acc = self.accuracy(y_pred, y_true)
        class_acc = self.class_wise_accuracy(y_pred, y_true)
        
        metrics = {
            'dice_mean': dice_scores.mean().item(),
            'dice_per_class': dice_scores.cpu().numpy(),
            'iou_mean': iou_scores.mean().item(),
            'iou_per_class': iou_scores.cpu().numpy(),
            'accuracy': acc,
            'accuracy_per_class': class_acc.cpu().numpy()
        }
        
        return metrics

class MetricsLogger:
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_dice': [],
            'val_dice': [],
            'train_iou': [],
            'val_iou': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def update(self, metrics: dict, phase: str):
        """Update metrics history with new values."""
        if phase == 'train':
            self.metrics_history['train_loss'].append(metrics.get('loss', 0))
            self.metrics_history['train_dice'].append(metrics.get('dice_mean', 0))
            self.metrics_history['train_iou'].append(metrics.get('iou_mean', 0))
            self.metrics_history['train_acc'].append(metrics.get('accuracy', 0))
            if 'learning_rate' in metrics:
                self.metrics_history['learning_rate'].append(metrics['learning_rate'])
        elif phase == 'val':
            self.metrics_history['val_loss'].append(metrics.get('loss', 0))
            self.metrics_history['val_dice'].append(metrics.get('dice_mean', 0))
            self.metrics_history['val_iou'].append(metrics.get('iou_mean', 0))
            self.metrics_history['val_acc'].append(metrics.get('accuracy', 0))
    
    def get_current_metrics(self) -> dict:
        """Get the most recent metrics."""
        current_metrics = {}
        for key, value in self.metrics_history.items():
            if value:  # If the list is not empty
                current_metrics[key] = value[-1]
        return current_metrics
    
    def get_best_metrics(self) -> dict:
        """Get the best metrics achieved."""
        best_metrics = {}
        for key, value in self.metrics_history.items():
            if value:  # If the list is not empty
                if 'loss' in key:
                    best_metrics[f'best_{key}'] = min(value)
                else:
                    best_metrics[f'best_{key}'] = max(value)
        return best_metrics
