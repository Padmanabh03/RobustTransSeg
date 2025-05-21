import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F


# Import our modules
from models.unet3d import UNet3D
from models.losses import CombinedLoss
from data.data_generator import get_data_loaders, DataAugmentation
from utils.metrics import SegmentationMetrics, MetricsLogger
from utils.visualization import BrainTumorVisualizer

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_validation_loss = float('inf') if mode == 'min' else float('-inf')
    
    def __call__(self, validation_loss):
        if self.mode == 'min':
            if validation_loss < self.min_validation_loss - self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
                return True  # Save model
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                return False
        else:  # mode == 'max'
            if validation_loss > self.min_validation_loss + self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
                return True  # Save model
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
                return False

def train_model(
    # Data parameters
    data_root="BraTS2020_TrainingData/input_data_128",
    batch_size=2,
    num_workers=4,
    
    # Model parameters
    in_channels=3,
    num_classes=4,
    base_filters=16,
    
    # Training parameters
    epochs=100,
    learning_rate=0.0001,
    device='cuda',
    
    # Augmentation parameters
    p_flip=0.5,
    p_rotate=0.5,
    
    # Early stopping parameters
    patience=7,
    min_delta=0.001,
    
    # Class weights for loss function
    class_weights=None,
    
    # Output parameters
    output_dir='outputs',
    experiment_name=None
):
    """Train the 3D UNet model with early stopping and visualization."""
    
    # Create output directories
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = os.path.join(output_dir, experiment_name)
    model_dir = os.path.join(experiment_dir, 'models')
    vis_dir = os.path.join(experiment_dir, 'visualizations')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set up data loaders
    transform = DataAugmentation(p_flip=p_flip, p_rotate=p_rotate)
    train_loader, val_loader, test_loader = get_data_loaders(
        train_img_dir=os.path.join(data_root, "train/images"),
        train_mask_dir=os.path.join(data_root, "train/masks"),
        val_img_dir=os.path.join(data_root, "val/images"),
        val_mask_dir=os.path.join(data_root, "val/masks"),
        test_img_dir=os.path.join(data_root, "test/images"),
        test_mask_dir=os.path.join(data_root, "test/masks"),
        batch_size=batch_size,
        num_workers=num_workers,
        transform=transform
    )
    
    # Initialize model, loss, and optimizer
    model = UNet3D(in_channels=in_channels, num_classes=num_classes, base_filters=base_filters)
    model = model.to(device)
    
    criterion = CombinedLoss(class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    # Initialize metrics and early stopping
    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics_logger = MetricsLogger()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    visualizer = BrainTumorVisualizer(save_dir=vis_dir)
    
    # Training loop
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        train_metrics = []
        
        for images, masks in tqdm(train_loader, desc="Training"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, torch.argmax(masks, dim=1))
            
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                batch_metrics = metrics.evaluate_batch(
                    F.softmax(outputs, dim=1),
                    torch.argmax(masks, dim=1)
                )
                batch_metrics['loss'] = loss.item()
                train_metrics.append(batch_metrics)
                train_losses.append(loss.item())
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, torch.argmax(masks, dim=1))
                
                batch_metrics = metrics.evaluate_batch(
                    F.softmax(outputs, dim=1),
                    torch.argmax(masks, dim=1)
                )
                batch_metrics['loss'] = loss.item()
                val_metrics.append(batch_metrics)
                val_losses.append(loss.item())
        
        # Calculate average metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        train_dice = np.mean([m['dice_mean'] for m in train_metrics])
        val_dice = np.mean([m['dice_mean'] for m in val_metrics])
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        metrics_logger.update(
            {
                'loss': train_loss,
                'dice_mean': train_dice,
                'learning_rate': optimizer.param_groups[0]['lr']
            },
            phase='train'
        )
        metrics_logger.update(
            {
                'loss': val_loss,
                'dice_mean': val_dice
            },
            phase='val'
        )
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save training curves
        visualizer.plot_training_curves(
            metrics_logger.metrics_history,
            save_path=f'training_curves_epoch_{epoch+1}.png'
        )
        
        # Early stopping check
        if early_stopping(val_loss):
            print("Saving best model...")
            model.save_model(
                best_model_path,
                epoch=epoch,
                optimizer=optimizer,
                loss=val_loss
            )
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model for testing
    print("\nLoading best model for testing...")
    model, _ = UNet3D.load_model(best_model_path, device)
    model.eval()
    
    # Test phase
    test_metrics = []
    all_images = []
    all_masks = []
    all_predictions = []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            predictions = F.softmax(outputs, dim=1)
            
            batch_metrics = metrics.evaluate_batch(predictions, torch.argmax(masks, dim=1))
            test_metrics.append(batch_metrics)
            
            # Store some examples for visualization
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_predictions.append(predictions.cpu())
    
    # Calculate and print test metrics
    test_dice = np.mean([m['dice_mean'] for m in test_metrics])
    test_iou = np.mean([m['iou_mean'] for m in test_metrics])
    print(f"\nTest Results:")
    print(f"Dice Score: {test_dice:.4f}")
    print(f"IoU Score: {test_iou:.4f}")
    
    # Save test results
    results_path = os.path.join(experiment_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Results:\n")
        f.write(f"Dice Score: {test_dice:.4f}\n")
        f.write(f"IoU Score: {test_iou:.4f}\n")
    
    # Visualize some test predictions
    for i in range(min(5, len(all_images))):  # Visualize up to 5 examples
        visualizer.plot_slices(
            all_images[i][0],  # Take first image from batch
            all_predictions[i][0],  # Take first prediction from batch
            save_path=f'test_prediction_{i+1}.png'
        )
        
        visualizer.create_3d_animation(
            all_images[i][0],
            all_predictions[i][0],
            save_path=f'test_prediction_{i+1}.gif'
        )
    
    print(f"\nTraining completed. Results saved in {experiment_dir}")
    return model, metrics_logger

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train model
    model, metrics_logger = train_model(
        device=device,
        epochs=100,
        batch_size=2,
        learning_rate=0.0001,
        patience=7
    )
