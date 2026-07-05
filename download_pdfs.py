"""
download_pdfs.py
----------------
Interactive script to browse, download, and upload/register PDFs for Document Understanding.
Supports financial, commercial, scientific, legal, and medical categories.
"""

import os
import sys
import csv
import shutil
import requests
from tabulate import tabulate

# Add root folder to sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from algorithms.config import DATA_DIR

REQUEST_TIMEOUT_SECONDS = 30
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

CATEGORIES = ["commercial", "financial", "legal", "medical", "scientific"]

# ── Helper Functions ──────────────────────────────────────────────────────────

def get_csv_path(category: str) -> str:
    """Return absolute path to a category's CSV metadata file."""
    if category == "commercial":
        return os.path.join(DATA_DIR, category, "commercial_pdfs.csv")
    return os.path.join(DATA_DIR, category, f"{category}_pdfs.csv")

def load_csv(csv_path: str) -> list[dict]:
    """Load metadata rows from the CSV file."""
    if not os.path.isfile(csv_path):
        print(f"\n[ERROR] CSV file not found at: {csv_path}")
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_csv(csv_path: str, rows: list[dict], fieldnames: list[str]) -> None:
    """Save metadata rows back to the CSV file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def download_file(url: str, destination_path: str) -> tuple[bool, str]:
    """Download a file with retry logic."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
            if response.status_code != 200:
                return False, f"HTTP {response.status_code}"
            
            with open(destination_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            size_kb = os.path.getsize(destination_path) // 1024
            return True, f"{size_kb} KB"
        except Exception as e:
            if attempt < RETRY_ATTEMPTS:
                print(f"    [Retry {attempt}] Failed ({e}). Retrying...")
            else:
                return False, str(e)
    return False, "All retries failed"

def download_pdf_flow(category: str):
    """Interactive flow to download PDFs in a category."""
    csv_path = get_csv_path(category)
    rows = load_csv(csv_path)
    if not rows:
        print("\nNo items found in this category's CSV.")
        return

    dest_folder = os.path.join(DATA_DIR, category)
    os.makedirs(dest_folder, exist_ok=True)

    while True:
        print(f"\n=== Category: {category.upper()} ===")
        table_data = []
        for idx, row in enumerate(rows, 1):
            doc_id = row.get("id", "")
            doc_type = row.get("doc_type", "")
            url = row.get("download_url", "")
            
            pdf_path = os.path.join(dest_folder, f"{doc_id}.pdf")
            status = "✅ Downloaded" if os.path.isfile(pdf_path) else "❌ Missing"
            table_data.append([idx, doc_id, doc_type, status, url[:40] + "..."])

        print(tabulate(table_data, headers=["#", "ID", "Type", "Status", "Download URL"], tablefmt="fancy_grid"))
        print("\nOptions:")
        print("  [A] Download ALL missing PDFs")
        print("  [#] Enter a number to download/redownload a specific PDF")
        print("  [B] Go Back")

        choice = input("\nChoose an option: ").strip().lower()
        if choice == 'b':
            break
        elif choice == 'a':
            print("\nStarting download pipeline...")
            success_count = 0
            for row in rows:
                doc_id = row["id"].strip()
                url = row["download_url"].strip()
                pdf_path = os.path.join(dest_folder, f"{doc_id}.pdf")
                
                if os.path.isfile(pdf_path):
                    continue
                
                print(f"\n[DL] Downloading {doc_id}...")
                success, msg = download_file(url, pdf_path)
                if success:
                    print(f"  [OK] Saved to {pdf_path} ({msg})")
                    success_count += 1
                else:
                    print(f"  [FAIL] {msg}")
                    if os.path.isfile(pdf_path):
                        os.remove(pdf_path)
            print(f"\nFinished: Downloaded {success_count} new file(s).")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(rows):
                    row = rows[idx]
                    doc_id = row["id"].strip()
                    url = row["download_url"].strip()
                    pdf_path = os.path.join(dest_folder, f"{doc_id}.pdf")
                    
                    print(f"\n[DL] Downloading {doc_id}...")
                    success, msg = download_file(url, pdf_path)
                    if success:
                        print(f"  [OK] Saved to {pdf_path} ({msg})")
                    else:
                        print(f"  [FAIL] {msg}")
                        if os.path.isfile(pdf_path):
                            os.remove(pdf_path)
                else:
                    print("\nInvalid index number.")
            except ValueError:
                print("\nInvalid option. Please try again.")

def upload_local_pdf_flow():
    """Interactive flow to copy a local PDF and register it in a category CSV."""
    print("\n=== IMPORT LOCAL PDF ===")
    pdf_path = input("Enter the absolute path of your local PDF file: ").strip().strip('"')
    if not os.path.isfile(pdf_path) or not pdf_path.lower().endswith(".pdf"):
        print("[ERROR] Invalid PDF file path.")
        return

    print("\nSelect the destination subclass:")
    for idx, cat in enumerate(CATEGORIES, 1):
        print(f"  [{idx}] {cat.capitalize()}")
    
    try:
        cat_choice = int(input("\nEnter choice (1-5): "))
        if not (1 <= cat_choice <= 5):
            print("[ERROR] Invalid choice.")
            return
        category = CATEGORIES[cat_choice - 1]
    except ValueError:
        print("[ERROR] Invalid choice.")
        return

    doc_id = input("\nEnter a unique ID for this PDF (e.g. MyDoc_01): ").strip()
    if not doc_id:
        print("[ERROR] ID cannot be empty.")
        return

    doc_type = input("Enter document type/sub-type (e.g. invoice, manual): ").strip()

    # Create category directory and copy
    dest_dir = os.path.join(DATA_DIR, category)
    os.makedirs(dest_dir, exist_ok=True)
    dest_pdf_path = os.path.join(dest_dir, f"{doc_id}.pdf")

    shutil.copy2(pdf_path, dest_pdf_path)
    print(f"\n[OK] Copied file to: {dest_pdf_path}")

    # Append to CSV
    csv_path = get_csv_path(category)
    rows = load_csv(csv_path)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = ["id", "doc_type", "download_url"]

    # Check if ID already exists
    if any(r.get("id") == doc_id for r in rows):
        print("[WARNING] An entry with this ID already exists in the CSV. The file was copied, but CSV entry was not duplicated.")
        return

    new_row = {"id": doc_id, "doc_type": doc_type, "download_url": ""}
    # Preserve other fields if present in CSV
    for f in fieldnames:
        if f not in new_row:
            new_row[f] = ""

    rows.append(new_row)
    save_csv(csv_path, rows, fieldnames)
    print(f"[OK] Registered entry in {os.path.basename(csv_path)}")

# ── Main Entry Point ──────────────────────────────────────────────────────────

def main():
    while True:
        print("\n" + "=" * 50)
        print("       DOCUMENT UNDERSTANDING - PDF MANAGER")
        print("=" * 50)
        print("Choose an action:")
        print("  [1] Browse / Download PDFs from category CSVs")
        print("  [2] Import a local PDF into the project")
        print("  [3] Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        if choice == '3':
            print("\nGoodbye!")
            break
        elif choice == '2':
            upload_local_pdf_flow()
        elif choice == '1':
            while True:
                print("\nSelect Category:")
                for idx, cat in enumerate(CATEGORIES, 1):
                    print(f"  [{idx}] {cat.capitalize()}")
                print("  [B] Back to main menu")
                
                cat_choice = input("\nEnter choice: ").strip().lower()
                if cat_choice == 'b':
                    break
                try:
                    idx = int(cat_choice) - 1
                    if 0 <= idx < len(CATEGORIES):
                        download_pdf_flow(CATEGORIES[idx])
                    else:
                        print("\nInvalid selection.")
                except ValueError:
                    print("\nInvalid option.")
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
