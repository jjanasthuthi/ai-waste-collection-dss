"""
01_download_data.py
-------------------
Downloads and extracts the Ultra Sound Garbage Bin Sensor dataset from Zenodo.

Dataset: https://zenodo.org/records/14988663
Expected: ~484 container CSV files (corrected fill sensor readings)

Usage:
    python src/01_download_data.py

Outputs:
    data/raw/   — all files from the Zenodo record, extracted in-place
"""

import os
import sys
import zipfile
import tarfile
import requests

ZENODO_RECORD_ID = "14988663"
ZENODO_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch_record_metadata():
    """Fetch Zenodo record metadata and return the list of file entries."""
    print(f"Fetching Zenodo record metadata: {ZENODO_API_URL}")
    resp = requests.get(ZENODO_API_URL, timeout=30)
    resp.raise_for_status()
    record = resp.json()
    files = record.get("files", [])
    if not files:
        # Newer Zenodo API layout
        files = record.get("metadata", {}).get("files", [])
    print(f"  Found {len(files)} file(s) in record.")
    return files


def download_file(url, dest_path):
    """Download a single file from `url` to `dest_path` with progress reporting."""
    print(f"  Downloading: {os.path.basename(dest_path)}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                f.write(chunk)
                downloaded += len(chunk)
        if total:
            print(f"    {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB")
        else:
            print(f"    {downloaded / 1e6:.1f} MB downloaded")


def extract_archive(path, dest_dir):
    """Extract a zip or tar archive into dest_dir."""
    if zipfile.is_zipfile(path):
        print(f"  Extracting zip: {os.path.basename(path)}")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(dest_dir)
        return True
    elif tarfile.is_tarfile(path):
        print(f"  Extracting tar: {os.path.basename(path)}")
        with tarfile.open(path, "r:*") as tf:
            tf.extractall(dest_dir)
        return True
    return False


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    files = fetch_record_metadata()
    if not files:
        print("ERROR: No files found in the Zenodo record. Check the record ID or API response.")
        sys.exit(1)

    for file_entry in files:
        # Zenodo file entries have 'key' (filename) and 'links.self' (download URL)
        filename = file_entry.get("key") or file_entry.get("filename", "unknown")
        links = file_entry.get("links", {})
        download_url = links.get("self") or links.get("download")
        if not download_url:
            print(f"  WARNING: No download URL for {filename}, skipping.")
            continue

        dest_path = os.path.join(RAW_DATA_DIR, filename)
        if os.path.exists(dest_path):
            print(f"  Already exists, skipping: {filename}")
        else:
            download_file(download_url, dest_path)

        # Extract archives immediately after download
        if filename.endswith((".zip", ".tar.gz", ".tgz", ".tar.bz2", ".tar")):
            extract_archive(dest_path, RAW_DATA_DIR)

    # Summarise what landed in raw/
    all_files = []
    for root, _, fnames in os.walk(RAW_DATA_DIR):
        for fname in fnames:
            all_files.append(os.path.join(root, fname))
    csv_files = [f for f in all_files if f.endswith(".csv")]
    print(f"\nDownload complete.")
    print(f"  Total files in data/raw: {len(all_files)}")
    print(f"  CSV files found:         {len(csv_files)}")


if __name__ == "__main__":
    main()
