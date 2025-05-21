import os
import argparse
import zipfile

def extract_training(zip_path: str, output_dir: str) -> None:
    """
    Extract only the training subset from a BRATS archive.

    Args:
        zip_path: Path to the .zip archive.
        output_dir: Directory where training files will be extracted.
    """
    TRAIN_PREFIX = "BraTS2020_TrainingData/"
    with zipfile.ZipFile(zip_path, 'r') as archive:
        members = [m for m in archive.infolist()
                   if m.filename.startswith(TRAIN_PREFIX) and not m.is_dir()]
        print(f"Found {len(members)} training files to extract...")
        for member in members:
            # Compute target path
            relative_path = os.path.relpath(member.filename, TRAIN_PREFIX)
            target_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            with archive.open(member) as source, open(target_path, 'wb') as target:
                target.write(source.read())
        print("Extraction of training data completed.")

def main():
    zip_path = "C:/Users/padma/OneDrive/Desktop/Python/RobustTransSeg/archive.zip"
    output_dir = "training_output"

    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    os.makedirs(output_dir, exist_ok=True)

    extract_training(zip_path, output_dir)

if __name__ == "__main__":
    main()
