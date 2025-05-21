# data_split.py
import os
import splitfolders   # pip install split-folders

# ------------------------------------------------------------------
# Paths
input_folder  = r'BraTS2020_TrainingData/input_data_3channels/'
output_folder = r'BraTS2020_TrainingData/input_data_128/'
# ------------------------------------------------------------------

# Check that the input exists
if not os.path.isdir(input_folder):
    raise FileNotFoundError(f"Input folder not found: {input_folder}")

# Create the output directory if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Split: 70 % train, 15 % val, 15 % test
splitfolders.ratio(
    input_folder,
    output      = output_folder,
    seed        = 42,
    ratio       = (0.7, 0.15, 0.15),
    group_prefix=None,   # change only if you need grouping by common prefix
    move        = True   # Move files instead of copying to save disk space
)

print("Dataset successfully split into train/val/test.")
