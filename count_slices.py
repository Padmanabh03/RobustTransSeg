import nibabel as nib
import os

# Define the path to the dataset
DATASET_PATH = 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'

# Get the first patient folder
patient_folders = sorted(os.listdir(DATASET_PATH))
first_patient = patient_folders[0]  # First patient folder

# Load the first image (using flair modality as an example)
first_image_path = os.path.join(DATASET_PATH, first_patient, f"{first_patient}_flair.nii")
first_image = nib.load(first_image_path).get_fdata()

# Get the shape of the image
image_shape = first_image.shape

print(f"Number of slices in the first image: {image_shape[2]}")
print(f"Full image dimensions (height × width × slices): {image_shape[0]} × {image_shape[1]} × {image_shape[2]}")
