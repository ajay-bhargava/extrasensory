import os
import modal

####
# Constants
####

BUCKET_NAME = "with-context-datasets"
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

prepare_dataset_image = (
    modal.Image.debian_slim()
    .apt_install(["tree", "curl"])
    .pip_install(
        "pandas",
        "numpy",
        "matplotlib",
        "tsai",
        "gzip"
    )
)