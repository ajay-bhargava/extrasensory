import os
import modal

####
# Constants
####

BUCKET_NAME = "with-context-datasets"
BASE_URL = "http://extrasensory.ucsd.edu/data/primary_data_files"
FILE_NAME = "ExtraSensory.per_uuid_features_labels.zip"
AWS_PATH = "/mnt/datasets/extrasensory"

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

download_image = (
    modal.Image.debian_slim()
    .apt_install(["tree", "curl"])
    .pip_install(
        "aiohttp",
        "rich",
        "tqdm",
        "asyncio"
    )
    .add_local_python_source("extrasensory")
)

####
# Imports
####

with download_image.imports():
    import subprocess
    from rich import print
    import os
    from pathlib import Path
    import sys
    import shutil
    from extrasensory.download.utils.monitors import start_monitoring_disk_space
    from extrasensory.download.utils.utilities import download_file, extract_zip, copy_concurrent

####
# Process
####

app = modal.App(
    name="extrasensory-download",
    image=download_image,
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
def download_extrasensory_data():
    if os.path.exists("/tmp/extrasensory"):
        shutil.rmtree("/tmp/extrasensory")
    os.makedirs("/tmp/extrasensory/extracted", exist_ok=True)
    
    print("🔄 Starting download...")
    try:
        start_monitoring_disk_space()
        subprocess.run(
            ["tree", AWS_PATH],
            check=True,
        )
        download_path = download_file(
            dest_dir="/tmp/extrasensory",
            base_url=BASE_URL,
            file=FILE_NAME,
        )
        extraction_path = extract_zip(
            zip_path=download_path,
            dest=Path("/tmp/extrasensory/extracted"),
        )
        copy_concurrent(
            src=extraction_path,
            dest=Path(AWS_PATH),
        )
        print("✅ Function complete. Files copied to AWS.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise e

