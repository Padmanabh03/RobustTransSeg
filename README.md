# RobustTransSeg

**Implementation and Comparison of CNN and Transformer-Based Architectures for Brain Tumor Segmentation**

## Overview
RobustTransSeg is a novel medical image segmentation model that integrates Convolutional Neural Networks (CNNs) and Transformer-based architectures to leverage both local and global features. This hybrid dual-encoder design provides robust segmentation outputs for complex and noisy medical imaging data. The project is built and tested using the BRATS dataset, focusing on brain tumor segmentation tasks.

## Features
- **Hybrid Dual-Encoder Architecture**:
  - CNN Encoder for local spatial feature extraction.
  - Transformer Encoder (Swin Transformer) for global contextual understanding.
- **Feature Fusion**:
  - Combines local and global features to create a unified representation.
- **Decoder with Skip Connections**:
  - Refines fused features and incorporates hierarchical features from the encoder.
- **Robust Training Mechanism**:
  - DiceCELoss for balancing class imbalance and voxel-wise accuracy.
  - Gradient scaling and learning rate scheduling for stable training.

## Dataset
This project uses the BRATS (Brain Tumor Segmentation) dataset, which contains multi-modal MRI scans with the following modalities:
- FLAIR
- T1-weighted
- T1-weighted with contrast enhancement (T1c)
- T2-weighted

The dataset includes voxel-wise annotations for the following classes:
- Background
- Edema
- Non-enhancing Tumor
- Enhancing Tumor

## Model Architecture
- **CNN Encoder**:
  - 4 convolutional blocks with down-sampling.
  - Outputs feature maps capturing local spatial details.
- **Transformer Encoder**:
  - Swin Transformer-based architecture.
  - Outputs hierarchical feature maps capturing global context.
- **Feature Fusion**:
  - Combines CNN and Transformer feature maps.
  - Outputs a fused representation for the decoder.
- **Decoder**:
  - 4 upsampling blocks with skip connections from the CNN encoder.
  - Produces high-resolution voxel-wise segmentation maps.

## Training Details
- **Loss Function**:
  - DiceCELoss: A combination of Dice Loss and Cross-Entropy Loss.
    ```
    Loss = λ_dice · Dice Loss + λ_ce · Cross-Entropy Loss
    ```
  - λ_dice = 0.5, λ_ce = 0.5

- **Optimizer**:
  - AdamW with learning rate = 1e-4

- **Scheduler**:
  - StepLR: Reduces learning rate by half every 50 epochs

- **Metrics**:
  - Dice Metric for evaluating segmentation performance.

## Results
- **Dice Score**: Approximately 0.613, highlighting room for improvement in segmentation performance.
- **Training and Validation Loss**: Steady decrease across epochs, indicating effective learning.

## Challenges and Future Work
- **Challenges**:
  - Computational bottlenecks due to the dual-encoder architecture.
  - Dataset imbalance affecting performance on rare classes like non-enhancing tumors.

- **Future Work**:
  - Increase output channel depth to capture richer feature representations.
  - Implement advanced loss functions like focal loss to address class imbalance.
  - Optimize memory usage and explore curriculum learning strategies.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Padmanabh03/RobustTransSeg.git
   cd RobustTransSeg
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare data:
   ```bash
   python data_preprocessing.py --input_dir path/to/brats2020
   python data_split.py
   ```

4. Train models:
   ```bash
   # For CNN approach
   python train.py

   # For Transformer+CNN approach (Coming Soon)
   python train_dual.py
   ```

5. Evaluate:
   ```bash
   python evaluate.py --model_path path/to/model
   ```

## Related Work
This implementation builds upon several established works in medical image segmentation:
- **3D UNet**: Volumetric segmentation architecture (Çiçek et al., 2016)
- **UNETR**: Transformers for medical image segmentation (Hatamizadeh et al., 2022)
- **SwinUNETR**: Hierarchical transformers (Tang et al., 2022)

## Future Updates
- Results from the Transformer+CNN approach
- Comparative analysis between both approaches
- Extended visualization examples
- Detailed ablation studies

## Acknowledgements
- BraTS 2020 dataset organizers
- Authors of UNETR, SwinUNETR, and other foundational works
- PyTorch and MONAI communities

## License
This project is licensed under the MIT License. See `LICENSE` for details.
