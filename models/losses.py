import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        
    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        
        if self.weight is None:
            self.weight = torch.ones(num_classes).to(logits.device)
            
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Calculate Dice score for each class
        dice_scores = []
        for i in range(num_classes):
            intersection = (probs[:, i] * targets_one_hot[:, i]).sum((1, 2, 3))
            union = probs[:, i].sum((1, 2, 3)) + targets_one_hot[:, i].sum((1, 2, 3))
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice * self.weight[i])
            
        return 1 - torch.mean(torch.stack(dice_scores))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits, targets):
        # Convert targets to one-hot encoding
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets.long(), num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Apply softmax to logits
        probs = F.softmax(logits, dim=1)
        
        # Calculate focal loss
        pt = (targets_one_hot * probs).sum(1) + 1e-10
        ce_loss = -torch.log(pt)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(weight=class_weights)
        self.focal_loss = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal
