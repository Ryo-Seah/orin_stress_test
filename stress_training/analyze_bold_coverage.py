import os
import pandas as pd

BOLD_ROOT = "/Users/RyoSeah/Desktop/Stress_Detection/BOLD_public"
ANNOTATIONS_FILE = os.path.join(BOLD_ROOT, "annotations", "train.csv")  # Change to val.csv or test.csv as needed

def main():
    annotations_df = pd.read_csv(ANNOTATIONS_FILE, header=None)
    print(f"Total annotations in {ANNOTATIONS_FILE}: {len(annotations_df)}")

    found = 0
    missing = 0
    missing_examples = []

    for idx, row in annotations_df.iterrows():
        video_path = row[0]  # e.g. 003/IzvOYVMltkI.mp4/0114.mp4
        # Convert to joint file path
        video_dir, segment_file = video_path.rsplit('/', 1)
        segment_id = segment_file.replace('.mp4', '.npy')
        joints_file = os.path.join(BOLD_ROOT, "joints", video_dir, segment_id)
        if os.path.exists(joints_file):
            found += 1
        else:
            missing += 1
            if len(missing_examples) < 10:
                missing_examples.append(joints_file)

    print(f"Annotations with matching joint file: {found}")
    print(f"Annotations missing joint file: {missing}")
    print("Sample missing joint files:")
    for f in missing_examples:
        print("  -", f)

if __name__ == "__main__":
    main()