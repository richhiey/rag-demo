from __future__ import annotations

import argparse
import shutil

from common import DB_DIR, LIVE_DIR, get_vector_store, load_config, load_source_documents, split_documents
from fetch_sources import refresh_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Load, split, embed, and index lesson documents.")
    parser.add_argument("--reset", action="store_true", help="Rebuild the local Chroma directory.")
    parser.add_argument("--refresh", action="store_true", help="Fetch the current live Nationalpark Schwarzwald pages before indexing.")
    args = parser.parse_args()
    load_config()

    if args.refresh or not list(LIVE_DIR.glob("*.md")):
        refresh_sources()

    if args.reset and DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    documents = load_source_documents()
    chunks = split_documents(documents)
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

    print(f"Loaded {len(documents)} live source documents")
    print(f"Created {len(chunks)} chunks with overlap")
    print(f"Indexed {len(chunks)} chunks in {DB_DIR}")
    if chunks:
        preview = chunks[0]
        print("Preview:")
        print(f"source_id={preview.metadata.get('source_id')}")
        print(f"source_url={preview.metadata.get('source_url')}")
        print(preview.page_content[:240].replace("\n", " ") + "...")


if __name__ == "__main__":
    main()
