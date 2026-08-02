from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = DEMO_DIR / "data" / "sources.json"


@dataclass(frozen=True)
class SourceDefinition:
    source_id: str
    title: str
    url: str


def load_source_definitions(path: Path = SOURCE_CONFIG) -> list[SourceDefinition]:
    raw_sources = json.loads(path.read_text(encoding="utf-8"))
    return [SourceDefinition(**source) for source in raw_sources]
