import pandas as pd
import os
import numpy as np

splits = ['train', 'val', 'test']
annotation_root = 'BOLD_public/annotations'
joint_root = 'BOLD_public/joints'

for split in splits:
    file_path = os.path.join(annotation_root, f"{split}.csv")
    df = pd.read_csv(file_path, header=None)
    print(f"{split}.csv shape: {df.shape}")
    print(f"{split}.csv columns: {df.shape[1]}")
    print(f"First row: {df.iloc[0].values}\n")
    missing_count = df.isnull().sum().sum()
    print(f"Missing values in {split}.csv: {missing_count}\n")
    # Check corresponding .npy joint files for correct pose-point format
    print(f"Checking joint .npy files for {split} split...")
    invalid_count = 0  # count files not matching 18 pose features
    for idx, row in df.iterrows():
        video_path = row[0]
        try:
            video_dir, segment_file = video_path.rsplit('/', 1)
            npy_file = segment_file.replace('.mp4', '.npy')
            npy_path = os.path.join(joint_root, video_dir, npy_file)
            if not os.path.exists(npy_path):
                # missing file; skip but do not count
                continue
            arr = np.load(npy_path)
            # Expect shape (N_frames, 56): 2 metadata + 18 joints×3
            if arr.ndim != 2 or arr.shape[1] != 56:
                print(f"Invalid shape {arr.shape} in {npy_path}")
                invalid_count += 1
        except Exception as e:
            print(f"Error checking {video_path}: {e}")
            # skip errors without counting
    print(f"Found {invalid_count} files without 18 pose features in {split} split\n")