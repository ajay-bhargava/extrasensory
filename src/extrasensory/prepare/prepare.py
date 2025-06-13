import os
import modal
from concurrent.futures import ThreadPoolExecutor, as_completed

####
# Constants
####

BUCKET_NAME = "with-context-datasets"
AWS_PATH = "/mnt/datasets/extrasensory"
OUTPUT_PATH = "/mnt/prepared/extrasensory"
WINDOW_SIZE = 10
OVERLAP_SIZE = 5

####
# Credentials
####

CREDENTIALS = modal.Secret.from_name(
    "aws-secret",
    environment_name="main",
    required_keys=[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
)

####
# Image
####

prepare_dataset_image = (
    modal.Image.debian_slim()
    .apt_install(["tree", "curl"])
    .pip_install(
        "pandas",
        "numpy",
        "matplotlib",
        "rich",
    )
    .add_local_python_source("extrasensory")
)

####
# Imports
####

with prepare_dataset_image.imports():
    import pandas as pd
    import numpy as np
    import subprocess
    from pathlib import Path
    import gzip
    import matplotlib.pyplot as plt
    from rich import print
    from extrasensory.prepare.utils.graph import participant_continuity_context
    from extrasensory.prepare.utils.massage import split_extrasensory_dataframe, mark_transition_windows, find_nonoverlapping_contiguous_samples


####
# App
####

app = modal.App(
    name="extrasensory-prepare",
    image=prepare_dataset_image,
)

def process_single_file(file, output_path):
    """Process a single participant file and return dataset and labels."""
    strip_name = file.name.replace(".features_labels.csv.gz", "")
    
    try:
        with gzip.open(file, "rt") as f: # type: ignore
            df = pd.read_csv(f) # type: ignore
    except Exception as e:
        print(f"❌ Error reading file {file}: {e}")
        return None

    try:
        X, Y, M, timestamps, feature_names, label_names = split_extrasensory_dataframe(df)

        print(f"👤 Processing User: {strip_name} ...")

        Y_T, label_names_transition = mark_transition_windows(Y, 10, 11, WINDOW_SIZE, label_names, ["before", "after", "transition"])

        print(f"🔄 Marked transition windows...")

        labels_to_display = ["OR_indoors", "OR_outside", "before", "after", "transition"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        figure, axis = participant_continuity_context(timestamps, Y_T, label_names_transition, labels_to_display, colors, strip_name)

        print(f"🔄 Plotted participant continuity context...")

        # Ensure EDA directory exists
        os.makedirs(f"{output_path}/eda", exist_ok=True)
        figure.savefig(f"{output_path}/eda/{strip_name}.png")
        plt.close(figure)  # Close figure to free memory

        transition_outside_mask = (
            ((Y_T[:, label_names_transition.index("before")] == True) | 
            (Y_T[:, label_names_transition.index("after")] == True)) &
            (Y_T[:, label_names_transition.index("OR_outside")] == True)
        )

        transition_outside_indices = np.where(transition_outside_mask)[0]

        transition_inside_mask = (
            ((Y_T[:, label_names_transition.index("before")] == True) | 
            (Y_T[:, label_names_transition.index("after")] == True)) &
            (Y_T[:, label_names_transition.index("OR_indoors")] == True)
        )

        transition_inside_indices = np.where(transition_inside_mask)[0]
        
        print(f"🔄 Created Masks for transition windows.")
        
        inside_windows = find_nonoverlapping_contiguous_samples(transition_inside_indices, window_size=OVERLAP_SIZE)
        outside_windows = find_nonoverlapping_contiguous_samples(transition_outside_indices, window_size=OVERLAP_SIZE)

        print(f"🔄 Found {len(inside_windows)} inside windows and {len(outside_windows)} outside windows.")
        
        # Check if either window list is empty
        if len(inside_windows) == 0 or len(outside_windows) == 0:
            print(f"⚠️  Skipping user {strip_name} - insufficient windows (inside: {len(inside_windows)}, outside: {len(outside_windows)}).")
            return None
            
        sensor_samples_inside = []
        for enum, item in enumerate(inside_windows):
            sensor_samples_inside.append(X[item])

        sensor_samples_outside = []
        for enum, item in enumerate(outside_windows):
            sensor_samples_outside.append(X[item])
            
        # Fuse Dataset as Sample x Features
        inside_array = np.stack([arr.T for arr in sensor_samples_inside], axis=0)
        outside_array = np.stack([arr.T for arr in sensor_samples_outside], axis=0)

        print(f"🔄 Fused Dataset as Sample x Features.")
        
        dataset = np.concatenate([inside_array, outside_array], axis=0)
        labels = np.concatenate([np.zeros(len(inside_windows), dtype=int), np.ones(len(outside_windows), dtype=int)])
        
        print(f"✅ Successfully processed user {strip_name}")
        return dataset, labels
        
    except Exception as e:
        print(f"❌ Error processing user {strip_name}: {e}")
        return None


@app.function(
    volumes={
        "/mnt": modal.CloudBucketMount(
            bucket_name=BUCKET_NAME,
            secret=CREDENTIALS,
        )
    },
    timeout=60 * 60 * 3,
)
def prepare_extrasensory_dataset():
    if os.path.exists(AWS_PATH):
        subprocess.run(["tree", AWS_PATH], check=True)
    else:
        raise FileNotFoundError(f"AWS_PATH {AWS_PATH} does not exist")
    
    files = list(Path(AWS_PATH).glob("*.csv.gz"))
    print(f"📁 Found {len(files)} files to process")

    # Process files concurrently
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all files for processing
        future_to_file = {executor.submit(process_single_file, file, OUTPUT_PATH): file for file in files}
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    print(f"📊 Collected results from {file.name}")
            except Exception as e:
                print(f"❌ Failed to process {file.name}: {e}")
    
    # Check if we have any valid results
    if not results:
        print("❌ No valid datasets were produced from any files.")
        return
    
    # Separate datasets and labels
    datasets, labels = zip(*results)
    
    # Combine all datasets and labels
    dataset_combined = np.concatenate(datasets, axis=0)
    labels_combined = np.concatenate(labels, axis=0)
    
    print(f"✅ Combined Dataset Produced - Shape: {dataset_combined.shape}, Labels: {labels_combined.shape}")
    
    print(f"💾 Saving Dataset...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    np.save(f"{OUTPUT_PATH}/dataset.npy", dataset_combined)
    np.save(f"{OUTPUT_PATH}/labels.npy", labels_combined)
    
    print(f"✅ Dataset Saved Successfully!")
    print(f"📊 Final dataset shape: {dataset_combined.shape}")
    print(f"📊 Final labels shape: {labels_combined.shape}")
    print(f"📊 Class distribution: {np.bincount(labels_combined)}")
    
    
    
        