# -*- coding: utf-8 -*-
"""RobustTransSeg_v2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XMmS5_UDfJ9tb3o-nTryafXupLJBLZkx

# Installing libraries
"""

# Install required libraries
!pip install torch torchvision torchaudio
!pip install monai==1.3.2
!pip install timm
!pip install segmentation-models-pytorch
!pip install albumentations

"""# Importing Libraries"""

# Import standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Import torchvision for image handling
import torchvision.transforms as transforms

# Import MONAI for medical image processing
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,Spacingd, Orientationd, ScaleIntensityRanged,
    CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, EnsureTyped,NormalizeIntensityd,Lambdad, ResizeWithPadOrCropd
)

# Import timm for Transformer models
import timm

# Import segmentation_models_pytorch for additional utilities
import segmentation_models_pytorch as smp

# Import Albumentations for advanced augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Verify imports
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"MONAI Version: {monai.__version__}")
print(f"timm Version: {timm.__version__}")
print(f"Segmentation Models PyTorch Version: {smp.__version__}")

"""# Downloading the Dataset"""

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/

import os
import tarfile

# Define the path to the .tar fileconten in Google Drive
drive_tar_path = '/content/drive/MyDrive/Task01_BrainTumour.tar'  # Update if different

# Define the extraction directory
extract_dir = '/content'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the .tar file
with tarfile.open(drive_tar_path, 'r') as tar:
    tar.extractall(path=extract_dir)

print("Extraction completed.")

! ls /content/Task01_BrainTumour

import os

# Define paths to images and labels
images_dir = '/content/Task01_BrainTumour/imagesTr'
labels_dir = '/content/Task01_BrainTumour/labelsTr'

# List the contents of images and labels directories
print("Images Directory Contents:")
print(os.listdir(images_dir)[:])  # Display first 5 files
print(len(os.listdir(images_dir)))

print("\nLabels Directory Contents:")
print(os.listdir(labels_dir)[:])  # Display first 5 files
print(len(os.listdir(labels_dir)))

"""# Exploring the json file"""

import json
import os

base_dir = '/content/Task01_BrainTumour'


# Define the path to dataset.json
dataset_json_path = '/content/Task01_BrainTumour/dataset.json'

# Load dataset.json
with open(dataset_json_path, 'r') as f:
    dataset_info = json.load(f)

# Display the keys in dataset.json
print("Keys in dataset.json:", dataset_info.keys())

# Explore the content under each key
for key in dataset_info:
    print(f"\nKey: {key}")
    print(dataset_info[key])

"""# Data Preprocessing"""

from monai.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the base directory where the dataset is extracted
base_dir = '/content/Task01_BrainTumour'  # Ensure this matches your directory name

# Paths to subdirectories
images_tr_dir = os.path.join(base_dir, 'imagesTr')
labels_tr_dir = os.path.join(base_dir, 'labelsTr')
images_ts_dir = os.path.join(base_dir, 'imagesTs')

# Path to dataset.json
dataset_json_path = os.path.join(base_dir, 'dataset.json')

# Load dataset.json
with open(dataset_json_path, 'r') as f:
    dataset_info = json.load(f)

# Inspect keys and sample entries
print("Keys in dataset.json:", dataset_info.keys())
print("\nSample Training Entries:")
for entry in dataset_info['training'][:5]:
    print(entry)

"""# Filtering out hidden files"""

def filter_hidden_files(file_list):
    return [file for file in file_list if not file.startswith('._')]

# List and filter image and label files
all_images = filter_hidden_files(os.listdir(images_tr_dir))
all_labels = filter_hidden_files(os.listdir(labels_tr_dir))

print(f"Total images (filtered): {len(all_images)}")
print(f"Total labels (filtered): {len(all_labels)}")

import logging

def make_abs_paths(file_list, base_dir):
    """
    Convert relative paths to absolute paths for image-label pairs.

    Args:
        file_list (list): List of dictionaries with 'image' and 'label' keys.
        base_dir (str): Base directory path.

    Returns:
        list: List of dictionaries with absolute 'image' and 'label' paths.
    """
    abs_files = []
    missing_files = []
    for item in file_list:
        try:
            rel_image = item['image'].replace('./', '').strip()
            rel_label = item['label'].replace('./', '').strip()
        except TypeError:
            # If item is not a dict, log and skip
            logging.error(f"Invalid item format (expected dict): {item}")
            continue

        # Construct absolute paths
        abs_image = os.path.join(base_dir, rel_image)
        abs_label = os.path.join(base_dir, rel_label)

        # Handle labels without '_seg' suffix if necessary
        if not os.path.exists(abs_label):
            # Attempt to find label without '_seg'
            alt_abs_label = abs_label.replace('_seg', '')
            if os.path.exists(alt_abs_label):
                abs_label = alt_abs_label
                print(f"Adjusted label path for {abs_image} to {abs_label}")
            else:
                print(f"Label file does not exist for {abs_image}: {abs_label}")
                missing_files.append(abs_image)
                continue  # Skip this pair

        # Check if image exists
        if not os.path.exists(abs_image):
            print(f"Image file does not exist: {abs_image}")
            missing_files.append(abs_image)
            continue  # Skip this pair

        # Append to the list if both files exist
        abs_files.append({'image': abs_image, 'label': abs_label})

    if missing_files:
        print(f"\nTotal missing files: {len(missing_files)}")
        for file in missing_files:
            print(f"Missing: {file}")
    else:
        print("\nAll image-label pairs are present.")

    return abs_files

def make_image_paths(file_list, base_dir):
    """
    Convert relative paths to absolute paths for images without labels.

    Args:
        file_list (list): List of image filenames or relative paths (strings).
        base_dir (str): Base directory path.

    Returns:
        list: List of dictionaries with absolute 'image' paths.
    """
    abs_files = []
    missing_files = []
    for filename in file_list:
        if not isinstance(filename, str):
            logging.error(f"Invalid filename format (expected string): {filename}")
            continue

        # Remove any leading './' from the filename
        rel_image = filename.replace('./', '').strip()

        # Construct the absolute path without adding 'imagesTs' again
        abs_image = os.path.join(base_dir, rel_image)

        # Check if the image file exists
        if not os.path.exists(abs_image):
            print(f"Image file does not exist: {abs_image}")
            missing_files.append(abs_image)
            continue  # Skip this file

        # Append to the list if the file exists
        abs_files.append({'image': abs_image})

    if missing_files:
        print(f"\nTotal missing test files: {len(missing_files)}")
        for file in missing_files:
            print(f"Missing: {file}")
    else:
        print("\nAll test image files are present.")

    return abs_files

# Apply the function to training set
train_files = make_abs_paths(dataset_info.get('training', []), base_dir)

# Apply the function to test set
test_files = make_image_paths(dataset_info.get('test', []), base_dir)

print(f"\nNumber of training samples after path conversion: {len(train_files)}")
print(f"Number of test samples after path conversion: {len(test_files)}")

"""# Splitting Training data into training and validation"""

# Perform an 80-20 split for training and validation
train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)

print(f"Number of training samples after split: {len(train_files)}")
print(f"Number of validation samples: {len(val_files)}")

"""# Training Transforms (With Augmentation)"""

def map_labels(x):
    # Map label 4 to 3
    # This assumes you are working with BraTS labels
    x[x == 4] = 3
    return x

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    Lambdad(keys=["label"], func=map_labels),

    # Perform data augmentation that might change shape
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
    RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),

    # Now perform cropping to ensure consistent size
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(128, 128, 64),
        pos=1,
        neg=1,
        num_samples=2,
        image_key="image",
        image_threshold=0,
    ),

    # Final step: ensure that no matter what, the result is (128,128,64).
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 64)),

    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    Lambdad(keys=["label"], func=map_labels),

    # Validation might not need augmentation, but ensure uniform shape
    ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(128, 128, 64)),

    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

# train_transforms = Compose([
#     LoadImaged(keys=["image", "label"]),
#     EnsureChannelFirstd(keys=["image", "label"]),  # Adds channel dimension if missing
#     Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
#     Orientationd(keys=["image", "label"], axcodes="RAS"),
#     ScaleIntensityRanged(
#         keys=["image"],
#         a_min=-175,
#         a_max=250,
#         b_min=0.0,
#         b_max=1.0,
#         clip=True,
#     ),
#     CropForegroundd(keys=["image", "label"], source_key="image"),
#     RandCropByPosNegLabeld(
#         keys=["image", "label"],
#         label_key="label",
#         spatial_size=(64,64,64),
#         pos=1,
#         neg=1,
#         num_samples=4,
#         image_key="image",
#         image_threshold=0,
#     ),
#     RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
#     RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
#     EnsureTyped(keys=["image", "label"]),
# ])

"""# Validation Transform (Without Augmentation)"""

# from monai.transforms import ResizeWithPadOrCropd

# val_transforms = Compose([
#     LoadImaged(keys=["image", "label"]),
#     EnsureChannelFirstd(keys=["image", "label"]),
#     Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
#     Orientationd(keys=["image", "label"], axcodes="RAS"),
#     ScaleIntensityRanged(
#         keys=["image"],
#         a_min=-175,
#         a_max=250,
#         b_min=0.0,
#         b_max=1.0,
#         clip=True,
#     ),
#     CropForegroundd(keys=["image", "label"], source_key="image"),
#     ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(64,64,64)),
#     EnsureTyped(keys=["image", "label"]),
# ])

"""# Creating MONAI Datasets and Dataloaders"""

from monai.data import Dataset as MonaiDataset

train_ds = MonaiDataset(data=train_files, transform=train_transforms)
val_ds = MonaiDataset(data=val_files, transform=val_transforms)

# # Create Datasets and DataLoaders
# train_ds = Dataset(data=train_files, transform=train_transforms)
# val_ds = Dataset(data=val_files, transform=val_transforms)

# Define DataLoader parameters optimized for Colab Pro's T4 GPU
batch_size = 1                # Minimum possible to reduce memory usage
num_workers = 2               # Reduced to minimize memory overhead
pin_memory = True             # Enable pinned memory for faster GPU transfers
persistent_workers = True     # Keeps workers alive between epochs to reduce overhead
prefetch_factor = 2           # Lower to reduce memory usage

from monai.data import list_data_collate

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    collate_fn=list_data_collate
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    collate_fn=list_data_collate
)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Fetch a batch from the training DataLoader
try:
    batch = next(iter(train_loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    print(f'Image batch shape: {images.shape}')  # Expected: [B, C, H, W, D]
    print(f'Label batch shape: {labels.shape}')  # Expected: [B, C, H, W, D]
except RuntimeError as e:
    print("RuntimeError:", e)

# import matplotlib.pyplot as plt

# # Fetch a batch from the training DataLoader
# batch = next(iter(train_loader))
# images = batch["image"].to(device)
# labels = batch["label"].to(device)

# # Select the first sample in the batch
# img = images[0, 0, :, :, 64].cpu().numpy()  # First modality (e.g., T1) and middle slice
# lbl = labels[0, 0, :, :, 64].cpu().numpy()

# # Plot the image and label
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# axs[0].imshow(img, cmap='gray')
# axs[0].set_title('Image Slice (Modality 1)')
# axs[0].axis('off')

# axs[1].imshow(lbl, cmap='gray')
# axs[1].set_title('Label Slice')
# axs[1].axis('off')

# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, NormalizeIntensityd,
    EnsureTyped, Compose
)

# Example: minimal transforms just for loading and normalizing
visualize_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

# Create a small dataset and loader for visualization
vis_ds = Dataset(data=train_files, transform=visualize_transforms)
vis_loader = DataLoader(vis_ds, batch_size=1, shuffle=False)

# Fetch a few samples from the dataset
num_samples_to_show = 3
samples = []
for i, batch_data in enumerate(vis_loader):
    if i >= num_samples_to_show:
        break
    samples.append(batch_data)

# Let's assume we have 4 modalities: [FLAIR, T1, T1c, T2]
modality_names = ["FLAIR", "T1", "T1c", "T2"]

# Visualization of a random slice from each sample
for idx, batch_data in enumerate(samples):
    image = batch_data["image"][0].numpy()  # [C, D, H, W]
    label = batch_data["label"][0].numpy()  # [C, D, H, W], usually C=1 for segmentation labels

    # Select a middle slice for visualization
    depth = image.shape[1]
    mid_slice = depth // 2

    fig, axs = plt.subplots(1, len(modality_names)+1, figsize=(15, 5))
    fig.suptitle(f"Sample {idx+1}", fontsize=16)

    # Show each modality
    for m in range(len(modality_names)):
        axs[m].imshow(image[m, mid_slice, :, :], cmap="gray")
        axs[m].set_title(modality_names[m])
        axs[m].axis("off")

    # Show label overlay on top of one modality (e.g., FLAIR)
    # Assuming label is single-channel segmentation
    label_slice = label[0, mid_slice, :, :] if label.ndim == 4 else label[mid_slice, :, :]
    label_mask = np.ma.masked_where(label_slice == 0, label_slice)  # mask background

    axs[-1].imshow(image[0, mid_slice, :, :], cmap="gray") # base modality for overlay
    axs[-1].imshow(label_mask, cmap="jet", alpha=0.5)
    axs[-1].set_title("Label Overlay")
    axs[-1].axis("off")

    plt.tight_layout()
    plt.show()

# ------------------------
# Data Distribution Plots
# ------------------------
# We can plot histograms of intensity distributions for each modality across a few samples.

all_intensities = {m: [] for m in modality_names}

# Collect intensities from all samples and modalities
for batch_data in samples:
    image = batch_data["image"][0].numpy()  # [C, D, H, W]
    # Flatten out the volumes per modality and store
    for m_idx, m_name in enumerate(modality_names):
        intensities = image[m_idx].flatten()
        # Only take a subset to speed up if large images:
        intensities = intensities[intensities > 0]  # ignore zeros if background
        all_intensities[m_name].extend(intensities[:10000])  # sample up to 10k voxels

# Plot histograms
fig, axs = plt.subplots(1, len(modality_names), figsize=(15, 5))
fig.suptitle("Intensity Distribution per Modality", fontsize=16)
for i, m_name in enumerate(modality_names):
    axs[i].hist(all_intensities[m_name], bins=50, color='blue', alpha=0.7)
    axs[i].set_title(m_name)
    axs[i].set_xlabel("Intensity")
    axs[i].set_ylabel("Count")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    NormalizeIntensityd, EnsureTyped
)

# Example transform pipeline (adjust as needed for your dataset)
visualize_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
])

# Create a small dataset and loader for visualization
vis_ds = Dataset(data=train_files, transform=visualize_transforms)
vis_loader = DataLoader(vis_ds, batch_size=1, shuffle=False)

# Number of samples and slices to visualize
num_samples_to_show = 3
slices_to_show = 3  # number of slices per volume

samples = []
for i, batch_data in enumerate(vis_loader):
    if i >= num_samples_to_show:
        break
    samples.append(batch_data)

# Assuming we have 4 modalities: [FLAIR, T1, T1c, T2]
modality_names = ["FLAIR", "T1", "T1c", "T2"]

###############################################
# Visualizing multiple samples and slices
###############################################
for sample_idx, batch_data in enumerate(samples):
    image = batch_data["image"][0].numpy()  # shape: [C, D, H, W]
    label = batch_data["label"][0].numpy()  # shape: [C, D, H, W] usually C=1

    c, d, h, w = image.shape

    # Choose slices: start, middle, end
    depths = [d // 4, d // 2, 3 * d // 4]  # just an example; pick different slices
    fig, axs = plt.subplots(slices_to_show, len(modality_names)+1, figsize=(20, 4*slices_to_show))
    fig.suptitle(f"Sample {sample_idx+1}", fontsize=16)

    for si, slc in enumerate(depths):
        # Visualize each modality
        for m in range(len(modality_names)):
            axs[si, m].imshow(image[m, slc, :, :], cmap="gray")
            axs[si, m].set_title(f"{modality_names[m]} - Slice {slc}")
            axs[si, m].axis("off")

        # Label overlay on top of FLAIR (assuming index 0 is FLAIR)
        label_slice = label[0, slc, :, :] if label.ndim == 4 else label[slc, :, :]
        label_mask = np.ma.masked_where(label_slice == 0, label_slice)  # mask background
        axs[si, -1].imshow(image[0, slc, :, :], cmap="gray")
        axs[si, -1].imshow(label_mask, cmap="jet", alpha=0.5)
        axs[si, -1].set_title(f"Label Overlay - Slice {slc}")
        axs[si, -1].axis("off")

    plt.tight_layout()
    plt.show()

###############################################
# Data Distribution (Histograms) for Many Voxels
###############################################
all_intensities = {m: [] for m in modality_names}

# Collect intensities from all samples and modalities
for batch_data in samples:
    image = batch_data["image"][0].numpy()  # [C, D, H, W]
    for m_idx, m_name in enumerate(modality_names):
        intensities = image[m_idx].flatten()
        intensities = intensities[intensities > 0]  # ignore background/zeros if needed
        # Take more voxels to get a better distribution sample
        all_intensities[m_name].extend(intensities[:50000])  # up to 50k voxels

# Plot histograms for each modality
fig, axs = plt.subplots(1, len(modality_names), figsize=(15, 5))
fig.suptitle("Intensity Distribution per Modality", fontsize=16)
for i, m_name in enumerate(modality_names):
    axs[i].hist(all_intensities[m_name], bins=50, color='blue', alpha=0.7)
    axs[i].set_title(m_name)
    axs[i].set_xlabel("Intensity")
    axs[i].set_ylabel("Count")

plt.tight_layout()
plt.show()

"""# Required Libraries"""

from monai.networks.nets import SwinUNETR  # Example Transformer-based encoder
from monai.networks.layers import Conv
from monai.networks.blocks import Convolution, UpSample

"""```
Input Image
    |
[Dual Encoders]
    |          \
[CNN Encoder] [Transformer Encoder]
    |                   |
[Feature Maps]        [Global Features]
    |                   |
    |-------[Feature Fusion]-------
    |                   |
[Decoder with Skip Connections]
    |
Segmentation Output

```

# CNN Encoder
"""

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_channels=[16, 32, 64, 128]):
        super(CNNEncoder, self).__init__()
        layers = []
        prev_channels = in_channels
        for out_channels in feature_channels:
            layers.append(nn.Conv3d(prev_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))  # Downsampling
            prev_channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, nn.MaxPool3d):
                features.append(x)
        return features  # List of feature maps at different scales

"""# Vision Transformer Encoder"""

!wget -O model_swinvit.pt https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_size=24, img_size=(64, 64, 64)):
        super(TransformerEncoder, self).__init__()
        self.transformer = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,  # final segmentation channels
            feature_size=feature_size,
            use_checkpoint=True
        )

    def forward(self, x):
        # Extract intermediate features directly
        features = self.transformer.swinViT(x)  # returns a tuple/list of features at multiple resolutions

        # Print out all feature shapes to identify correct index
        for i, f in enumerate(features):
            print(f"features[{i}] shape:", f.shape)

        # Choose the feature map that matches the CNN's deepest layer spatial dimension
        # For example, if features[3] matches the CNN deepest scale, use that.
        transformer_deep_feature = features[3]
        return transformer_deep_feature

class TransformerEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_size=48, img_size=(64, 64, 64), pretrained_weights_path=None):
        """
        Transformer Encoder using Swin UNETR with optional pre-trained weights.

        Args:
        - in_channels (int): Number of input channels.
        - feature_size (int): Size of features extracted by the transformer.
        - img_size (tuple): Size of the input image.
        - pretrained_weights_path (str, optional): Path to the pre-trained weights for Swin UNETR.
        """
        super(TransformerEncoder, self).__init__()
        self.transformer = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=feature_size,  # final segmentation channels
            feature_size=feature_size,
            use_checkpoint=True
        )

        # Load pre-trained weights if provided
        if pretrained_weights_path:
            weight = torch.load(pretrained_weights_path)
            self.transformer.load_from(weights=weight)
            print(f"Loaded pre-trained weights from {pretrained_weights_path}")

    def forward(self, x):
        """
        Forward pass through the transformer encoder.

        Args:
        - x (torch.Tensor): Input tensor of shape [B, C, H, W, D].

        Returns:
        - torch.Tensor: Deep feature map extracted from the transformer.
        """
        features = self.transformer.swinViT(x)  # Extract intermediate features directly

        # Debug: Print feature shapes
        for i, f in enumerate(features):
            print(f"features[{i}] shape:", f.shape)

        # Select the deepest feature map
        transformer_deep_feature = features[3]  # Assuming features[3] is the deepest
        return transformer_deep_feature

# Specify the path to the pre-trained weights
pretrained_weights_path = "/content/model_swinvit.pt"

# Create the TransformerEncoder with pre-trained weights
transformer_encoder = TransformerEncoder(
    in_channels=4,
    feature_size=48,
    img_size=(64, 64, 64),
    pretrained_weights_path=pretrained_weights_path
).to(device)

# Test with a sample input tensor
example_input = torch.rand((1, 4, 64, 64, 64)).to(device)
output_features = transformer_encoder(example_input)
print(f"Output feature shape: {output_features.shape}")

"""# Fusing Features from both Encoders"""

# class FeatureFusion(nn.Module):
#     def __init__(self, cnn_channels, transformer_channels):
#         super(FeatureFusion, self).__init__()
#         self.conv = nn.Conv3d(cnn_channels + transformer_channels, transformer_channels, kernel_size=1)

#     def forward(self, cnn_features, transformer_features):
#         # Concatenate along the channel dimension
#         fused = torch.cat((cnn_features, transformer_features), dim=1)
#         fused = self.conv(fused)
#         return fused

class FeatureFusion(nn.Module):
    def __init__(self, cnn_channels=128, transformer_channels=192):
        super(FeatureFusion, self).__init__()
        # Now that we know total channels = 128 + 192 = 320
        self.conv = nn.Conv3d(cnn_channels + transformer_channels, 128, kernel_size=1)

    def forward(self, cnn_feature, transformer_feature):
        if cnn_feature.shape[2:] != transformer_feature.shape[2:]:
            transformer_feature = nn.functional.interpolate(
                transformer_feature, size=cnn_feature.shape[2:], mode='trilinear', align_corners=True
            )

        fused = torch.cat((cnn_feature, transformer_feature), dim=1)  # [B, 320, D, H, W]
        fused = self.conv(fused)  # [B, 128, D, H, W]
        return fused

"""# Decoder"""

# class Decoder(nn.Module):
#     def __init__(self, feature_channels=[128, 64, 32, 16], out_channels=1):
#         super(Decoder, self).__init__()
#         layers = []
#         prev_channels = feature_channels[0]
#         for out_c in feature_channels[1:]:
#             layers.append(
#                 nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#             )
#             layers.append(nn.Conv3d(prev_channels, out_c, kernel_size=3, padding=1))
#             layers.append(nn.BatchNorm3d(out_c))
#             layers.append(nn.ReLU(inplace=True))
#             prev_channels = out_c
#         layers.append(nn.Conv3d(prev_channels, out_channels, kernel_size=1))  # Final segmentation map
#         self.decoder = nn.Sequential(*layers)

#     def forward(self, x, skip_connections):
#         for idx in range(len(self.decoder)):
#             layer = self.decoder[idx]
#             if isinstance(layer, nn.Upsample):
#                 x = layer(x)
#                 # Retrieve the corresponding skip connection
#                 skip = skip_connections[-(idx//4 + 1)]  # Modify if necessary
#                 # Ensure spatial dimensions match
#                 if x.shape != skip.shape:
#                     skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=True)
#                 x = x + skip  # Simple addition; alternatively, concatenate
#             else:
#                 x = layer(x)
#         return x

class Decoder(nn.Module):
    def __init__(self, skip_channels=[128, 64, 32, 16], out_channels=1):
        super(Decoder, self).__init__()
        # Now add four upsample+conv stages corresponding to each skip connection.

        # Stage 1: from 4->8
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv3d(128 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # Stage 2: from 8->16
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv3d(64 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )

        # Stage 3: from 16->32
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            nn.Conv3d(32 + 16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        # Stage 4: from 32->64 (If you have another skip at a shallower level)
        # If you don't have another skip, just upsample and conv without skip:
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv4 = nn.Sequential(
            nn.Conv3d(16, 8, kernel_size=3, padding=1),  # or adjust based on your architecture
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv3d(8, out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        # This depends on how many skip connections you have
        # Adjust accordingly.

        # 1: from 4->8, use the deepest skip
        x = self.up1(x)
        skip = skip_connections[-1]
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)

        # 2: from 8->16
        x = self.up2(x)
        skip = skip_connections[-2]
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv2(x)

        # 3: from 16->32
        x = self.up3(x)
        skip = skip_connections[-3]
        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv3(x)

        # 4: from 32->64
        # If you have a fourth skip connection:
        # skip = skip_connections[-4]
        # interpolate if necessary, then concat
        # x = torch.cat([x, skip], dim=1)
        # Or if you don't have a fourth skip, just upsample:
        x = self.up4(x)
        x = self.conv4(x)

        x = self.final_conv(x)
        return x

"""# Assembling the Dual Encoder U-Net"""

class RobustTransSeg(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, img_size=(64, 64, 64)):
        super(RobustTransSeg, self).__init__()
        self.cnn_encoder = CNNEncoder(in_channels=in_channels, feature_channels=[16, 32, 64, 128])
        self.transformer_encoder = TransformerEncoder(in_channels=in_channels, feature_size=24, img_size=img_size)
        # self.fusion = FeatureFusion(cnn_channels=128, transformer_channels=24)  # Adjusted channels
        self.fusion = FeatureFusion(cnn_channels=128, transformer_channels=192)
        # self.decoder = Decoder(feature_channels=[128, 64, 32, 16], out_channels=out_channels)
        self.decoder = Decoder(skip_channels=[128, 64, 32, 16], out_channels=out_channels)


    def forward(self, x):
        # CNN Encoder
        cnn_features = self.cnn_encoder(x)  # List of feature maps at different scales

        # Transformer Encoder
        transformer_features = self.transformer_encoder(x)  # High-level global features

        # Fuse features
        fused = self.fusion(cnn_features[-1], transformer_features)  # Use the deepest CNN feature map

        # Decoder with skip connections
        output = self.decoder(fused, cnn_features[:-1])  # Exclude the last feature map used for fusion
        return output

"""# Initialization of the Model"""

print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))


# Initialize the model with reduced image size
# model = RobustTransSeg(in_channels=4, out_channels=1, img_size=(64, 64, 64)).to(device)
model = RobustTransSeg(in_channels=4, out_channels=1, img_size=(64, 64, 64))
print(model)

from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import decollate_batch
from monai.transforms import EnsureTyped
from tqdm import tqdm  # For progress bars
import logging
from torch.cuda.amp import GradScaler, autocast

# # Define the loss function
# criterion = DiceCELoss(to_onehot_y=False, softmax=False, include_background=True)
# # Define the optimizer
# optimizer = AdamW(model.parameters(), lr=1e-4)
# # Define the learning rate scheduler
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

import logging

# Configure logging before training
logging.basicConfig(
    level=logging.INFO,              # Set the minimum logging level to INFO
    format='%(asctime)s %(levelname)s: %(message)s',  # Format of the log messages
    datefmt='%Y-%m-%d %H:%M:%S'     # Optional: specify a date/time format
)
logger = logging.getLogger(__name__)

# architecture is defined as RobustTransSeg with out_channels=4
model = RobustTransSeg(in_channels=4, out_channels=4, img_size=(128, 128, 64)).to(device)

# For multi-class, use DiceCELoss with softmax=True, to_onehot_y=True
criterion = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    include_background=True,
    lambda_dice=0.5,
    lambda_ce=0.5
)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
# Optionally use a scheduler after you confirm stable training
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # less aggressive scheduling

dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
scaler = GradScaler()

num_epochs = 10
best_val_loss = float('inf')
accumulation_steps = 4

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    step = 0

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")):
        images, labels = batch["image"].to(device), batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += (loss.item() * accumulation_steps)

    epoch_loss /= (step + 1)
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    val_steps = 0
    dice_metric.reset()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, labels = batch["image"].to(device), batch["label"].to(device)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_steps += 1

            # For multi-class dice, we can directly feed softmaxed outputs into DiceMetric
            # The criterion already does softmax internally, but we can apply here again for clarity:
            softmax_preds = torch.softmax(outputs, dim=1)
            dice_metric(y_pred=softmax_preds, y=labels)

    val_loss /= val_steps
    dice_score = dice_metric.aggregate().item()
    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Dice Score: {dice_score:.4f}")

    scheduler.step()

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), '/content/drive/MyDrive/RobustTransSeg_best_model.pth')
        logger.info("Best model saved.")

    # Optional: save checkpoint at intervals
    if (epoch + 1) % 2 == 0:
        checkpoint_path = f"/content/drive/MyDrive/RobustTransSeg_epoch{epoch+1}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "dice_score": dice_score
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at epoch {epoch+1} with dice score {dice_score:.4f}")

    torch.cuda.empty_cache()

checkpoint = torch.load("RobustTransSeg_epoch5.pth")
print("Epoch:", checkpoint["epoch"])
print("Validation Loss:", checkpoint["val_loss"])
print("Dice Score:", checkpoint["dice_score"])





