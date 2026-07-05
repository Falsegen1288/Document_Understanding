"""Interactive PDF manifest, download, and local upload helpers."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent
CATEGORIES = ["commercial", "financial", "legal", "medical", "scientific"]
CSV_COLUMNS = ["id", "doc_type", "download_url"]


def csv_path_for(category: str) -> Path:
    """Return data/<category>/<category>_pdfs.csv."""
    category = category.lower().strip()
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Choose one of: {CATEGORIES}")
    return DATA_DIR / category / f"{category}_pdfs.csv"


def ensure_category(category: str) -> Path:
    """Create and return a category directory and CSV if missing."""
    category_dir = DATA_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path_for(category)
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
    return category_dir


def load_manifest(category: str) -> list[dict[str, str]]:
    """Load one category manifest."""
    csv_path = csv_path_for(category)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_manifest(category: str, rows: list[dict[str, str]]) -> None:
    """Write one category manifest."""
    ensure_category(category)
    with csv_path_for(category).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def list_pdfs(category: str) -> list[dict[str, Any]]:
    """Return manifest rows with local path/status attached."""
    rows = load_manifest(category)
    enriched = []
    for row in rows:
        doc_id = row.get("id", "").strip()
        path = DATA_DIR / category / f"{doc_id}.pdf"
        enriched.append({**row, "path": path, "downloaded": path.is_file()})
    return enriched


def get_pdf_path(
    category: str,
    doc_id: str | None = None,
    index: int = 0,
    must_exist: bool = False,
) -> Path:
    """Return the expected local path for a PDF in a category."""
    rows = load_manifest(category)
    if not rows:
        raise FileNotFoundError(f"No manifest rows found for category '{category}'.")
    if doc_id is None:
        if index < 0 or index >= len(rows):
            raise IndexError(f"Index {index} out of range for category '{category}'.")
        doc_id = rows[index]["id"]
    path = DATA_DIR / category / f"{doc_id}.pdf"
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"PDF is listed but not downloaded yet: {path}")
    return path


def add_url(category: str, doc_id: str, doc_type: str, download_url: str) -> None:
    """Append or replace a URL-backed PDF entry."""
    rows = [row for row in load_manifest(category) if row.get("id") != doc_id]
    rows.append({"id": doc_id, "doc_type": doc_type, "download_url": download_url})
    save_manifest(category, rows)


def upload_local_pdf(
    category: str,
    source_pdf: str | Path,
    doc_id: str | None = None,
    doc_type: str = "",
    add_to_manifest: bool = True,
    overwrite: bool = False,
) -> Path:
    """Copy a local PDF into data/<category> and optionally add it to the CSV."""
    category_dir = ensure_category(category)
    source = Path(source_pdf).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Local PDF not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {source}")

    doc_id = doc_id or source.stem
    destination = category_dir / f"{doc_id}.pdf"
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists. Pass overwrite=True to replace it.")
    shutil.copy2(source, destination)

    if add_to_manifest:
        add_url(category, doc_id, doc_type, "")
    return destination


def download_pdf(
    category: str,
    doc_id: str,
    overwrite: bool = False,
    timeout: int = 30,
    retries: int = 3,
) -> Path:
    """Download one manifest PDF into its category folder."""
    rows = load_manifest(category)
    row = next((item for item in rows if item.get("id") == doc_id), None)
    if row is None:
        raise KeyError(f"{doc_id} is not listed in {csv_path_for(category)}")
    url = row.get("download_url", "").strip()
    if not url:
        raise ValueError(f"{doc_id} has no download_url; it may be a local upload.")

    destination = ensure_category(category) / f"{doc_id}.pdf"
    if destination.exists() and not overwrite:
        return destination

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        handle.write(chunk)
            return destination
        except requests.RequestException as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2)
    if destination.exists():
        destination.unlink()
    raise RuntimeError(f"Failed to download {doc_id}: {last_error}")


def download_many(category: str, doc_ids: list[str] | None = None) -> list[Path]:
    """Download selected PDFs, or every URL-backed PDF in a category."""
    rows = load_manifest(category)
    wanted = set(doc_ids) if doc_ids else {row["id"] for row in rows if row.get("download_url")}
    return [download_pdf(category, doc_id) for doc_id in wanted]


def _print_table(rows: list[list[Any]], headers: list[str]) -> None:
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
        return
    print(" | ".join(headers))
    print("-" * 80)
    for row in rows:
        print(" | ".join(str(value) for value in row))


def _choose_category() -> str | None:
    print("\nCategories")
    _print_table([[i + 1, category] for i, category in enumerate(CATEGORIES)], ["#", "category"])
    raw = input("\nPick category number: ").strip()
    if not raw:
        return None
    try:
        return CATEGORIES[int(raw) - 1]
    except (ValueError, IndexError):
        print("Invalid category.")
        return None


def _show_category(category: str) -> list[dict[str, Any]]:
    rows = list_pdfs(category)
    print(f"\nPDFs in {category}")
    table_rows = []
    for index, row in enumerate(rows, start=1):
        table_rows.append(
            [
                index,
                row.get("id", ""),
                row.get("doc_type", ""),
                "yes" if row["downloaded"] else "no",
                row.get("download_url", "")[:70],
            ]
        )
    _print_table(table_rows, ["#", "id", "subclass", "downloaded", "url"])
    return rows


def interactive_main() -> None:
    """Interactive CLI for choosing, downloading, or uploading PDFs."""
    while True:
        print("\nPDF data manager")
        print("1. Show PDFs in a subclass")
        print("2. Download PDFs")
        print("3. Upload/copy a local PDF into a subclass")
        print("4. Add a URL to a subclass CSV")
        print("q. Quit")
        choice = input("\nChoice: ").strip().lower()

        if choice == "1":
            category = _choose_category()
            if category:
                _show_category(category)

        elif choice == "2":
            category = _choose_category()
            if not category:
                continue
            rows = _show_category(category)
            raw = input("\nEnter numbers to download, comma-separated, or 'all': ").strip().lower()
            if raw == "all":
                ids = [row["id"] for row in rows if row.get("download_url")]
            else:
                try:
                    indices = [int(part.strip()) - 1 for part in raw.split(",") if part.strip()]
                    ids = [rows[index]["id"] for index in indices]
                except (ValueError, IndexError):
                    print("Invalid selection.")
                    continue
            for doc_id in ids:
                try:
                    path = download_pdf(category, doc_id)
                    print(f"Downloaded: {path}")
                except Exception as exc:  # noqa: BLE001
                    print(f"Failed {doc_id}: {exc}")

        elif choice == "3":
            category = _choose_category()
            if not category:
                continue
            source = input("Path to local PDF: ").strip().strip('"')
            doc_id = input("Document id (blank = file name): ").strip() or None
            doc_type = input("Subclass/type label (optional): ").strip()
            try:
                path = upload_local_pdf(
                    category,
                    source,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    overwrite=False,
                )
                print(f"Uploaded/copied into: {path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Upload failed: {exc}")

        elif choice == "4":
            category = _choose_category()
            if not category:
                continue
            doc_id = input("Document id: ").strip()
            doc_type = input("Subclass/type label: ").strip()
            url = input("Download URL: ").strip()
            add_url(category, doc_id, doc_type, url)
            print(f"Added {doc_id} to {csv_path_for(category)}")

        elif choice in {"q", "quit", "exit"}:
            break
        else:
            print("Unknown choice.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or upload PDFs into data subclasses.")
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--download-all", action="store_true")
    parser.add_argument("--doc-id")
    parser.add_argument("--upload")
    parser.add_argument("--doc-type", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not any([args.category, args.download_all, args.doc_id, args.upload]):
        interactive_main()
        return
    if not args.category:
        print("--category is required in non-interactive mode.")
        sys.exit(1)
    if args.upload:
        print(upload_local_pdf(args.category, args.upload, args.doc_id, args.doc_type))
    elif args.download_all:
        for path in download_many(args.category):
            print(path)
    elif args.doc_id:
        print(download_pdf(args.category, args.doc_id))
    else:
        _show_category(args.category)


if __name__ == "__main__":
    main()
