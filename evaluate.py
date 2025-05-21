import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse

from models.unet3d import UNet3D
from data.data_generator import get_data_loaders
from utils.metrics import SegmentationMetrics
from utils.visualization import BrainTumorVisualizer

def evaluate_model(
    model_path: str,
    data_root: str = "BraTS2020_TrainingData/input_data_128",
    batch_size: int = 2,
    num_workers: int = 4,
    device: str = 'cuda',
    output_dir: str = 'evaluation_results',
    save_all_slices: bool = False
):
    """
    Evaluate a trained model on the test set and generate visualizations.
    
    Args:
        model_path: Path to the saved model checkpoint
        data_root: Root directory containing the dataset
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        device: Device to run evaluation on ('cuda' or 'cpu')
        output_dir: Directory to save evaluation results
        save_all_slices: If True, saves visualizations for all test samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, checkpoint = UNet3D.load_model(model_path, device)
    model.eval()
    
    # Create data loaders (we only need the test loader)
    _, _, test_loader = get_data_loaders(
        train_img_dir=os.path.join(data_root, "train/images"),
        train_mask_dir=os.path.join(data_root, "train/masks"),
        val_img_dir=os.path.join(data_root, "val/images"),
        val_mask_dir=os.path.join(data_root, "val/masks"),
        test_img_dir=os.path.join(data_root, "test/images"),
        test_mask_dir=os.path.join(data_root, "test/masks"),
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize metrics and visualizer
    metrics = SegmentationMetrics(num_classes=model.num_classes)
    visualizer = BrainTumorVisualizer(save_dir=output_dir)
    
    # Evaluation loop
    test_metrics = []
    class_wise_metrics = {
        'dice': [],
        'iou': [],
        'accuracy': []
    }
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions
            outputs = model(images)
            predictions = F.softmax(outputs, dim=1)
            
            # Calculate metrics
            batch_metrics = metrics.evaluate_batch(predictions, torch.argmax(masks, dim=1))
            test_metrics.append(batch_metrics)
            
            # Store class-wise metrics
            class_wise_metrics['dice'].append(batch_metrics['dice_per_class'])
            class_wise_metrics['iou'].append(batch_metrics['iou_per_class'])
            class_wise_metrics['accuracy'].append(batch_metrics['accuracy_per_class'])
            
            # Generate visualizations
            if save_all_slices or batch_idx < 5:  # Save first 5 samples or all if requested
                for i in range(images.size(0)):
                    sample_idx = batch_idx * batch_size + i
                    
                    # Save 2D slice visualizations
                    visualizer.plot_slices(
                        images[i].cpu(),
                        predictions[i].cpu(),
                        save_path=f'test_sample_{sample_idx}_slices.png'
                    )
                    
                    # Save 3D visualization
                    visualizer.create_3d_animation(
                        images[i].cpu(),
                        predictions[i].cpu(),
                        save_path=f'test_sample_{sample_idx}_3d.gif'
                    )
    
    # Calculate and save overall metrics
    overall_metrics = {
        'dice_score': np.mean([m['dice_mean'] for m in test_metrics]),
        'iou_score': np.mean([m['iou_mean'] for m in test_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in test_metrics])
    }
    
    # Calculate class-wise metrics
    class_names = ['background', 'necrotic', 'edema', 'enhancing']
    class_wise_metrics = {
        'dice': np.mean(class_wise_metrics['dice'], axis=0),
        'iou': np.mean(class_wise_metrics['iou'], axis=0),
        'accuracy': np.mean(class_wise_metrics['accuracy'], axis=0)
    }
    
    # Save metrics to file
    results_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("Overall Metrics:\n")
        f.write(f"Dice Score: {overall_metrics['dice_score']:.4f}\n")
        f.write(f"IoU Score: {overall_metrics['iou_score']:.4f}\n")
        f.write(f"Accuracy: {overall_metrics['accuracy']:.4f}\n\n")
        
        f.write("Class-wise Metrics:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"\n{class_name}:\n")
            f.write(f"  Dice Score: {class_wise_metrics['dice'][i]:.4f}\n")
            f.write(f"  IoU Score: {class_wise_metrics['iou'][i]:.4f}\n")
            f.write(f"  Accuracy: {class_wise_metrics['accuracy'][i]:.4f}\n")
    
    print("\nEvaluation Results:")
    print(f"Overall Dice Score: {overall_metrics['dice_score']:.4f}")
    print(f"Overall IoU Score: {overall_metrics['iou_score']:.4f}")
    print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"\nDetailed results saved to {results_path}")
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained 3D UNet model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model checkpoint")
    parser.add_argument("--data_root", type=str, default="BraTS2020_TrainingData/input_data_128",
                        help="Root directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to run evaluation on")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--save_all_slices", action="store_true",
                        help="Save visualizations for all test samples")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU instead.")
        args.device = "cpu"
    
    evaluate_model(
        model_path=args.model_path,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        output_dir=args.output_dir,
        save_all_slices=args.save_all_slices
    )
