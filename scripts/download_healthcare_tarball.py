import os
import sys
import requests

TARGET_PATH = os.path.abspath("external_benchmarks/UniDoc-Bench/pdf_archives/healthcare_pdfs.tar.gz")
URL = "https://huggingface.co/datasets/Salesforce/UniDoc-Bench/resolve/main/healthcare_pdfs.tar.gz"
EXPECTED_SIZE = 1282073224

def download_healthcare_tarball():
    os.makedirs(os.path.dirname(TARGET_PATH), exist_ok=True)
    initial_size = 0
    if os.path.exists(TARGET_PATH):
        initial_size = os.path.getsize(TARGET_PATH)
        if initial_size == EXPECTED_SIZE:
            print(f"Healthcare tarball is already complete: {initial_size} bytes.")
            return True
        elif initial_size > EXPECTED_SIZE:
            print(f"File size {initial_size} exceeds expected {EXPECTED_SIZE}, restarting download.")
            initial_size = 0

    headers = {}
    mode = "wb"
    if initial_size > 0:
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
        print(f"Resuming download from byte {initial_size}/{EXPECTED_SIZE}...")
    else:
        print(f"Starting fresh download of {EXPECTED_SIZE} bytes...")

    response = requests.get(URL, headers=headers, stream=True, timeout=30)
    if response.status_code not in (200, 206):
        print(f"Failed HTTP request: {response.status_code}")
        return False

    downloaded = initial_size
    with open(TARGET_PATH, mode) as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (50 * 1024 * 1024) < 1024 * 1024:
                    print(f"Progress: {downloaded / (1024*1024):.2f} MB / {EXPECTED_SIZE / (1024*1024):.2f} MB")

    final_size = os.path.getsize(TARGET_PATH)
    print(f"Download finished. Final file size: {final_size} bytes (Expected: {EXPECTED_SIZE}).")
    return final_size == EXPECTED_SIZE

if __name__ == "__main__":
    success = download_healthcare_tarball()
    if not success:
        sys.exit(1)
