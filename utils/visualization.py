import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import List, Tuple, Optional

class BrainTumorVisualizer:
    def __init__(self, save_dir: str = 'visualizations'):
        """
        Initialize the visualizer.
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Define colors for different tumor regions
        self.colors = {
            'background': [0, 0, 0, 0],      # Transparent
            'necrotic': [1, 0, 0, 0.7],      # Red
            'edema': [0, 1, 0, 0.7],         # Green
            'enhancing': [0, 0, 1, 0.7]      # Blue
        }
    
    def _prepare_data(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for visualization."""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
            
        # Ensure correct dimensions
        if image.ndim == 4:  # (C, H, W, D)
            image = image.transpose(1, 2, 3, 0)  # -> (H, W, D, C)
        if mask.ndim == 4:  # (C, H, W, D)
            mask = np.argmax(mask, axis=0)  # Convert one-hot to label
            
        return image, mask
    
    def plot_slices(self, image: np.ndarray, mask: np.ndarray, slice_nums: Optional[List[int]] = None,
                   save_path: Optional[str] = None):
        """
        Plot multiple slices of the brain with tumor overlay.
        Args:
            image: Image array of shape (H, W, D, C) or (C, H, W, D)
            mask: Mask array of shape (H, W, D) or (C, H, W, D)
            slice_nums: List of slice numbers to plot. If None, picks evenly spaced slices
            save_path: Path to save the plot
        """
        image, mask = self._prepare_data(image, mask)
        
        if slice_nums is None:
            num_slices = 6
            depth = image.shape[2]
            slice_nums = np.linspace(depth//4, 3*depth//4, num_slices, dtype=int)
        
        fig, axes = plt.subplots(2, len(slice_nums), figsize=(4*len(slice_nums), 8))
        plt.suptitle('Brain MRI Slices with Tumor Segmentation', fontsize=16)
        
        for i, slice_num in enumerate(slice_nums):
            # Plot original image (using T1CE channel)
            axes[0, i].imshow(image[:, :, slice_num, 1], cmap='gray')
            axes[0, i].set_title(f'Original (Slice {slice_num})')
            axes[0, i].axis('off')
            
            # Plot segmentation overlay
            axes[1, i].imshow(image[:, :, slice_num, 1], cmap='gray')
            for j, (name, color) in enumerate(self.colors.items()):
                if j > 0:  # Skip background
                    mask_region = (mask[:, :, slice_num] == j)
                    axes[1, i].imshow(np.ma.masked_where(~mask_region, mask_region),
                                    cmap=plt.matplotlib.colors.ListedColormap([color]),
                                    alpha=0.5)
            axes[1, i].set_title(f'Segmentation (Slice {slice_num})')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def create_3d_animation(self, image: np.ndarray, mask: np.ndarray, 
                          save_path: Optional[str] = None, fps: int = 10):
        """
        Create a rotating 3D animation of the tumor segmentation.
        Args:
            image: Image array of shape (H, W, D, C) or (C, H, W, D)
            mask: Mask array of shape (H, W, D) or (C, H, W, D)
            save_path: Path to save the animation
            fps: Frames per second for the animation
        """
        image, mask = self._prepare_data(image, mask)
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create surface plots for each tumor region
        surfaces = []
        for j, (name, color) in enumerate(self.colors.items()):
            if j > 0:  # Skip background
                mask_region = (mask == j)
                z, x, y = np.where(mask_region)
                if len(z) > 0:  # Only plot if region exists
                    scatter = ax.scatter(x, y, z, c=[color], alpha=0.5)
                    surfaces.append(scatter)
        
        def update(frame):
            ax.view_init(elev=20., azim=frame)
            return surfaces
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=np.arange(0, 360, 2),
                           interval=50, blit=True)
        
        if save_path:
            anim.save(os.path.join(self.save_dir, save_path), fps=fps)
            plt.close()
        else:
            plt.show()
    
    def plot_training_curves(self, metrics_history: dict, save_path: Optional[str] = None):
        """
        Plot training and validation curves.
        Args:
            metrics_history: Dictionary containing metrics history
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        plt.suptitle('Training Progress', fontsize=16)
        
        # Plot loss
        axes[0, 0].plot(metrics_history['train_loss'], label='Train')
        axes[0, 0].plot(metrics_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        
        # Plot Dice score
        axes[0, 1].plot(metrics_history['train_dice'], label='Train')
        axes[0, 1].plot(metrics_history['val_dice'], label='Validation')
        axes[0, 1].set_title('Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        
        # Plot IoU
        axes[1, 0].plot(metrics_history['train_iou'], label='Train')
        axes[1, 0].plot(metrics_history['val_iou'], label='Validation')
        axes[1, 0].set_title('IoU Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        
        # Plot accuracy
        axes[1, 1].plot(metrics_history['train_acc'], label='Train')
        axes[1, 1].plot(metrics_history['val_acc'], label='Validation')
        axes[1, 1].set_title('Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    # Test the visualizer
    visualizer = BrainTumorVisualizer()
    
    # Create dummy data
    image = np.random.rand(128, 128, 128, 3)  # (H, W, D, C)
    mask = np.random.randint(0, 4, (128, 128, 128))  # (H, W, D)
    
    # Test slice visualization
    visualizer.plot_slices(image, mask, save_path='test_slices.png')
    
    # Test 3D animation
    visualizer.create_3d_animation(image, mask, save_path='test_animation.gif')
    
    # Test training curves
    metrics = {
        'train_loss': np.random.rand(100),
        'val_loss': np.random.rand(100),
        'train_dice': np.random.rand(100),
        'val_dice': np.random.rand(100),
        'train_iou': np.random.rand(100),
        'val_iou': np.random.rand(100),
        'train_acc': np.random.rand(100),
        'val_acc': np.random.rand(100)
    }
    visualizer.plot_training_curves(metrics, save_path='test_curves.png')
