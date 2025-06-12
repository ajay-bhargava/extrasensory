import shlex
import shutil
import subprocess
import zipfile
from pathlib import Path



def download_file(
    dest_dir: str | Path = ".",
    base_url: str = "http://extrasensory.ucsd.edu/data/primary_data_files",
    file: str = "ExtraSensory.per_uuid_features_labels.zip"
) -> Path:
    """
    Download a single file using curl with HTTP/2 and parallel transfer options.
    
    Parameters
    ----------
    dest_dir : str | Path
        Where to save the file (directory is created if needed).
    base_url : str
        Base URL path that contains the file.
    file : str
        Filename to download (without any prefix).
    """
    dest_dir = Path(dest_dir).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Build the full URL and output path
    url = f"{base_url}/{file}"
    output_path = dest_dir / file
    
    # Build curl command with the specified options
    curl_cmd: list[str] = [
        "curl",
        "-Z",                       # run transfers in parallel (though single file here)
        "--http2",                  # try HTTP/2 multiplexing
        "--fail",
        "-L",                       # follow redirects
        "--retry", "5", "--retry-delay", "3",
        "-o", str(output_path),     # output file
        url                         # source URL
    ]
    
    print("»", " ".join(shlex.quote(t) for t in curl_cmd))   # debug
    subprocess.run(curl_cmd, check=True)
    print(f"✅ {file} downloaded successfully to {output_path}")
    return output_path


def extract_zip(zip_path: Path, dest: Path) -> Path:
    """
    Extract a zip file to a destination directory.
    """
    print(f"📂 extracting {zip_path} → {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    
    print("✅ Extraction complete")
    return dest

def copy_concurrent(src: Path, dest: Path, max_threads: int = 8) -> None:
    """
    Recursively copy *src* ➜ *dest* using a ThreadPool to maximise IO throughput.
    """
    from multiprocessing.pool import ThreadPool

    class MultithreadedCopier:
        def __init__(self, max_threads=8):
            self.pool = ThreadPool(max_threads)
            self.completed = 0
            self.errors = 0

        def copy(self, s, d):
            def on_success(_):
                self.completed += 1
                print(f"✓ {Path(s).relative_to(src)} ({self.completed} files)")
            
            def on_error(exc):
                self.errors += 1
            
            return self.pool.apply_async(
                shutil.copy2,
                args=(s, d),
                callback=on_success,
                error_callback=on_error,
            )

        def close(self):
            self.pool.close()
            self.pool.join()

    print(f"📁 Copying {src} → {dest}")
    copier = MultithreadedCopier(max_threads)
    
    try:
        shutil.copytree(src, dest, copy_function=copier.copy, dirs_exist_ok=True)
    finally:
        copier.close()