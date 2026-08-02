from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEMO_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = DEMO_DIR / "data"
LIVE_DIR = DATA_DIR / "live"
DB_DIR = DEMO_DIR / "chroma_db"
COLLECTION_NAME = "schwarzwald_trip_planner_live"


def load_config() -> None:
    load_dotenv(DEMO_DIR / ".env")


def _read_metadata(text: str, key: str) -> str:
    match = re.search(rf"^<!--\s*{re.escape(key)}:\s*(.*?)\s*-->\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def load_source_documents(data_dir: Path = LIVE_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        source_url = _read_metadata(raw, "source_url")
        source_id = _read_metadata(raw, "source_id") or path.stem
        source_title = _read_metadata(raw, "source_title") or path.stem
        fetched_at = _read_metadata(raw, "fetched_at")
        body = re.sub(r"^<!--.*?-->\s*$", "", raw, flags=re.MULTILINE).strip()
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "source_url": source_url,
                    "source_id": source_id,
                    "source_title": source_title,
                    "fetched_at": fetched_at,
                    "file": path.name,
                },
            )
        )
    return documents


def split_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def get_embeddings() -> HuggingFaceEmbeddings:
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(DB_DIR),
    )


def format_context(documents: Iterable[Document]) -> str:
    blocks = []
    for index, doc in enumerate(documents, start=1):
        blocks.append(
            f"[Source {index}: {doc.metadata.get('source_id', 'unknown')}]\n"
            f"URL: {doc.metadata.get('source_url', 'unknown')}\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)
