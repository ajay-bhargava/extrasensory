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
    import numpy as np
    import subprocess
    from pathlib import Path
    from rich import print
    from extrasensory.prepare.execution_thread import process_single_file

####
# App
####

app = modal.App(
    name="extrasensory-prepare",
    image=prepare_dataset_image,
)

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
    
    # Combine all datasets and labels into a single array (n_samples, n_features, timesteps) and labels (n_samples)
    dataset_combined = np.concatenate(datasets, axis=0)
    labels_combined = np.concatenate(labels, axis=0)
    
    # Remove features that are all np.nan
    nan_features = np.all(np.isnan(dataset_combined), axis=(0, 2))
    dataset_combined = dataset_combined[:, ~nan_features, :]
    
    # Transpose and remove the np.nan features that remain 
    dataset_combined = dataset_combined.transpose(0,2,1)
    dataset_combined = np.nan_to_num(dataset_combined, nan=0)
    
    print(f"💾 Saving Dataset...")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    np.save(f"{OUTPUT_PATH}/dataset.npy", dataset_combined)
    np.save(f"{OUTPUT_PATH}/labels.npy", labels_combined)
    
    print(f"✅ Dataset Saved Successfully!")
    print(f"📊 Final dataset shape: {dataset_combined.shape}")
    print(f"📊 Final labels shape: {labels_combined.shape}")
    print(f"📊 Class distribution: {np.bincount(labels_combined)}")
    
    
    
        