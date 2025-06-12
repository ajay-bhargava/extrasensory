import os
import s3fs
import tempfile
import gzip
import pandas as pd
import numpy as np

def explore_s3_bucket(fs, bucket_path):
    """
    Helper function to explore S3 bucket contents.
    
    Parameters:
    -----------
    fs : s3fs.S3FileSystem
        Filesystem object from mount_s3_bucket()
    bucket_path : str
        Path to explore (e.g., 'my-bucket/data/')
    Returns:
    --------
    List[str]: List of S3 paths (keys) for all items in the bucket_path
    """

    # List files and directories
    items = fs.ls(bucket_path, detail=True)
    s3_paths = []
    
    for item in items:
        s3_paths.append(item['Key'])
    
    return s3_paths


def mount_s3_bucket(
    bucket_name: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
):
    """
    Mount an S3 bucket with s3fs, optionally pulling creds from a `.env` file.
    """

    # ➋ If args weren't explicitly passed, fall back to env vars
    bucket_name          = bucket_name          or os.getenv("S3_BUCKET")
    aws_access_key_id    = aws_access_key_id    or os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")

    if not bucket_name:
        raise ValueError("Bucket name is required (argument or S3_BUCKET in .env).")

    # Clean bucket name (strip leading schema)
    bucket_name = bucket_name.replace("s3://", "")

    # ➌ Instantiate filesystem
    if aws_access_key_id and aws_secret_access_key:
        fs = s3fs.S3FileSystem(key=aws_access_key_id,
                               secret=aws_secret_access_key)
    else:
        # s3fs will fall back to regular AWS credential resolution chain
        fs = s3fs.S3FileSystem()

    # ➍ Verify access
    try:
        fs.ls(bucket_name)
        print(f"✓ Connected to s3://{bucket_name}")
    except Exception as exc:
        print(f"✗ Failed to connect: {exc}")
        return None

    return fs

def open_s3_file_locally(fs, s3_path, mode='rb') -> pd.DataFrame:
    """
    Download a file from S3 to a temporary location and open it.
    The file will be automatically cleaned up when closed.
    
    Parameters:
    -----------
    fs : s3fs.S3FileSystem
        S3 filesystem object
    s3_path : str
        Full S3 path to the file (e.g., 'my-bucket/path/to/file.csv')
    mode : str
        File open mode ('r' for text, 'rb' for binary)
    
    Returns:
    --------
    file : file-like object
        Opened file object that will be automatically cleaned up when closed
    
    Example:
    --------
    with open_s3_file_locally(fs, 'my-bucket/data.csv', 'r') as f:
        content = f.read()
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Download the file from S3
        fs.download(s3_path, temp_path)
        
        if s3_path.endswith(".csv.gz"):
            with gzip.open(temp_path, 'rt') as f:
                return pd.read_csv(f)
        else:
            return pd.read_csv(temp_path)
    except Exception as e:
        # Clean up the temporary file if there's an error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e

def cleanup_temp_file(file_obj):
    """
    Clean up the temporary file after use.
    """
    if hasattr(file_obj, 'name'):
        try:
            file_obj.close()
            os.unlink(file_obj.name)
        except Exception as e:
            print(f"Warning: Could not clean up temporary file: {e}")

