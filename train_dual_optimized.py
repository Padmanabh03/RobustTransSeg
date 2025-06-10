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
from models.transformer.dual_encoder_optimized import DualEncoderUNetOptimized, SingleEncoderUNet
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

def train_dual_model_optimized(
    # Data parameters
    data_root="BraTS2020_TrainingData/input_data_128",
    batch_size=4,  # Increased from 2 to 4 for better GPU utilization
    num_workers=2,  # Reduced from 4 to prevent CPU bottleneck
    
    # Model parameters - INCREASED FOR BETTER PERFORMANCE
    in_channels=3,
    num_classes=4,
    transformer_feature_size=48,  # Increased to match UNet3D capacity
    cnn_base_filters=16,  # Matching UNet3D base_filters
    
    # Training parameters
    epochs=100,  # Full 100 epochs
    learning_rate=0.001,  # Increased from 0.0001 for faster convergence
    device='cuda',
    
    # Augmentation parameters - REDUCED
    p_flip=0.3,  # Reduced from 0.5
    p_rotate=0.3,  # Reduced from 0.5
    
    # Early stopping parameters
    patience=15,  # Increased for better convergence
    min_delta=0.001,
    
    # Class weights for loss function
    class_weights=None,
    
    # Output parameters
    output_dir='outputs_dual',
    experiment_name=None,
    
    # Pretrained weights
    use_pretrained=True,
    
    # Performance optimizations
    use_amp=True,  # Use automatic mixed precision
    gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch size
    gradient_clip_value=1.0,  # Gradient clipping
    
    # Single encoder mode option
    use_single_encoder=False,  # Set to True to use only transformer encoder
):
    """Train the Dual Encoder U-Net model with optimizations for speed."""
    
    # Create output directories
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    experiment_dir = os.path.join(output_dir, experiment_name)
    model_dir = os.path.join(experiment_dir, 'models')
    vis_dir = os.path.join(experiment_dir, 'visualizations')
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Set up data loaders with optimization
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
    
    # Initialize model with reduced parameters
    if use_single_encoder:
        # Option to use only SwinUNETR without dual encoding
        model = SingleEncoderUNet(
            img_size=(128, 128, 128),
            in_channels=in_channels,
            out_channels=num_classes,
            feature_size=transformer_feature_size,
            use_pretrained=use_pretrained,
            use_gradient_checkpointing=True
        )
    else:
        model = DualEncoderUNetOptimized(
            img_size=(128, 128, 128),
            in_channels=in_channels,
            out_channels=num_classes,
            transformer_feature_size=transformer_feature_size,
            cnn_base_filters=cnn_base_filters,
            dropout_rates=[0.1, 0.1, 0.15, 0.15, 0.2],  # Reduced dropout
            use_pretrained=use_pretrained,
            use_cnn_encoder=True,  # Can set to False for even faster training
            use_gradient_checkpointing=True
        )
    
    model = model.to(device)
    
    # Use DistributedDataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    criterion = CombinedLoss(class_weights=class_weights)
    
    # Use AdamW with weight decay for better convergence
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Use OneCycleLR for faster convergence
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * 10,
        epochs=epochs,
        steps_per_epoch=len(train_loader) // gradient_accumulation_steps
    )
    
    # Initialize metrics and early stopping
    metrics = SegmentationMetrics(num_classes=num_classes)
    metrics_logger = MetricsLogger()
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)
    visualizer = BrainTumorVisualizer(save_dir=vis_dir)
    
    # Initialize GradScaler for automatic mixed precision
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    print(f"Training Optimized Model with:")
    print(f"- Model type: {'Single Encoder' if use_single_encoder else 'Dual Encoder'}")
    print(f"- Feature size: {transformer_feature_size}")
    print(f"- Batch size: {batch_size}")
    print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"- Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"- Mixed precision training: {use_amp}")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"- Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training loop
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_losses = []
        train_metrics = []
        optimizer.zero_grad()
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Automatic mixed precision
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, torch.argmax(masks, dim=1))
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, torch.argmax(masks, dim=1))
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
            
            # Calculate metrics (less frequently to save time)
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    batch_metrics = metrics.evaluate_batch(
                        F.softmax(outputs, dim=1),
                        torch.argmax(masks, dim=1)
                    )
                    batch_metrics['loss'] = loss.item() * gradient_accumulation_steps
                    train_metrics.append(batch_metrics)
                    train_losses.append(loss.item() * gradient_accumulation_steps)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_metrics = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, torch.argmax(masks, dim=1))
                else:
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
        
        # Log metrics
        current_lr = optimizer.param_groups[0]['lr']
        metrics_logger.update(
            {
                'loss': train_loss,
                'dice_mean': train_dice,
                'learning_rate': current_lr
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
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save training curves every 5 epochs
        if epoch % 5 == 0:
            visualizer.plot_training_curves(
                metrics_logger.metrics_history,
                save_path=f'training_curves_epoch_{epoch+1}.png'
            )
        
        # Early stopping check based on validation Dice (negative because we want to maximize)
        if early_stopping(-val_dice):
            print("Saving best model...")
            if hasattr(model, 'module'):  # DataParallel wrapper
                model.module.save_model(
                    best_model_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    loss=val_loss
                )
            else:
                model.save_model(
                    best_model_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    loss=val_loss
                )
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Save final training curves
    visualizer.plot_training_curves(
        metrics_logger.metrics_history,
        save_path='final_training_curves.png'
    )
    
    # Complete test evaluation
    print("\nPerforming complete test evaluation...")
    if hasattr(model, 'module'):
        test_model = model.module
    else:
        test_model = model
    
    test_model.eval()
    test_metrics = []
    test_predictions = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
                
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = test_model(images)
            else:
                outputs = test_model(images)
                
            predictions = F.softmax(outputs, dim=1)
            batch_metrics = metrics.evaluate_batch(predictions, torch.argmax(masks, dim=1))
            test_metrics.append(batch_metrics)
            
            # Save first few batches for visualization
            if i < 3:
                test_predictions.append({
                    'images': images.cpu(),
                    'masks': masks.cpu(),
                    'predictions': predictions.cpu()
                })
    
    # Calculate and print complete test metrics
    test_dice = np.mean([m['dice_mean'] for m in test_metrics])
    test_iou = np.mean([m['iou_mean'] for m in test_metrics])
    test_dice_per_class = np.mean([[m['dice_per_class'][i] for m in test_metrics] 
                                   for i in range(num_classes)], axis=1)
    
    print(f"\nComplete Test Results (All Batches):")
    print(f"Overall Dice Score: {test_dice:.4f}")
    print(f"Overall IoU Score: {test_iou:.4f}")
    print(f"Dice Scores per Class:")
    for i in range(num_classes):
        print(f"  Class {i}: {test_dice_per_class[i]:.4f}")
    
    # Generate test visualizations
    print("\nGenerating test visualizations...")
    for i, pred_data in enumerate(test_predictions):
        # Use first sample from batch
        image = pred_data['images'][0]
        mask = pred_data['masks'][0]
        prediction = pred_data['predictions'][0]
        
        # Convert prediction to class labels
        pred_labels = torch.argmax(prediction, dim=0).numpy()
        
        # Create slice visualization
        visualizer.plot_slices(
            image.numpy(),
            pred_labels,
            save_path=f'test_prediction_{i+1}.png'
        )
        
        # Create 3D animation for first test case
        if i == 0:
            visualizer.create_3d_animation(
                image.numpy(),
                pred_labels,
                save_path='test_prediction_3d.gif'
            )
    
    # Save results
    results_path = os.path.join(experiment_dir, 'training_summary.txt')
    with open(results_path, 'w') as f:
        f.write(f"Optimized Dual Encoder U-Net Training Summary:\n")
        f.write(f"Model type: {'Single Encoder' if use_single_encoder else 'Dual Encoder'}\n")
        f.write(f"Total epochs trained: {epoch + 1}\n")
        f.write(f"Best validation loss: {early_stopping.min_validation_loss:.4f}\n")
        f.write(f"Final validation dice: {val_dice:.4f}\n")
        f.write(f"Complete test dice (All batches): {test_dice:.4f}\n")
        f.write(f"\nModel Configuration:\n")
        f.write(f"- Transformer feature size: {transformer_feature_size}\n")
        f.write(f"- CNN base filters: {cnn_base_filters}\n")
        f.write(f"- Batch size: {batch_size}\n")
        f.write(f"- Gradient accumulation: {gradient_accumulation_steps}\n")
        f.write(f"- Mixed precision: {use_amp}\n")
        f.write(f"- Gradient clipping: {gradient_clip_value}\n")
    
    print(f"\nTraining completed. Results saved in {experiment_dir}")
    return model, metrics_logger

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device - check for CUDA, ROCm, or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Enable cuDNN optimizations only for NVIDIA GPUs
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"Using device: {device} (NVIDIA GPU)")
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        device = torch.device("cuda")  # ROCm also uses 'cuda' device name
        print(f"Using device: {device} (AMD GPU with ROCm)")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
        print("WARNING: Running on CPU will be slow. Consider installing PyTorch with GPU support.")
    
    # Adjust parameters based on device
    if device.type == 'cpu':
        # CPU-optimized parameters
        batch_size = 1  # Smaller batch size for CPU
        use_amp = False  # AMP doesn't help on CPU
        gradient_accumulation_steps = 4  # More accumulation to compensate for smaller batch
        num_workers = 0  # Single-threaded for CPU
        print("\nAdjusting parameters for CPU training:")
        print("- Batch size: 1 (reduced for CPU memory)")
        print("- AMP: Disabled (not beneficial on CPU)")
        print("- Gradient accumulation: 4 (to maintain effective batch size)")
        print("- Workers: 0 (single-threaded for CPU)")
        print("\nNOTE: Training on CPU will be significantly slower than GPU.")
        print("Consider using Google Colab with GPU or installing PyTorch with ROCm for AMD GPU.")
    else:
        # GPU parameters
        batch_size = 4
        use_amp = True
        gradient_accumulation_steps = 2
        num_workers = 2
    
    # Train optimized model
    model, metrics_logger = train_dual_model_optimized(
        device=device,
        epochs=100,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=0.001,
        patience=15,
        use_pretrained=True,
        use_amp=use_amp,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_single_encoder=False  # Set to True to use only transformer
    )
