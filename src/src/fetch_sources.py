from __future__ import annotations

import argparse
import re
import ssl
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import certifi

from source_registry import SourceDefinition, load_source_definitions


DEMO_DIR = Path(__file__).resolve().parents[1]
LIVE_DIR = DEMO_DIR / "data" / "live"
USER_AGENT = "rag-lesson-schwarzwald-day-planner/1.0 (educational demo)"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _safe_filename(source_id: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", source_id.lower()).strip("-") + ".md"


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("main") or soup.select_one("article") or soup
    for element in main(["script", "style", "nav", "aside", "header", "footer", "form"]):
        element.decompose()
    text = main.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_source(source: SourceDefinition, timeout: int = 20) -> str:
    request = Request(source.url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout, context=SSL_CONTEXT) as response:
        html = response.read().decode("utf-8", errors="replace")
    text = extract_main_text(html)
    if len(text) < 300:
        raise ValueError(f"The page returned too little readable text: {source.url}")
    return text


def write_cached_source(source: SourceDefinition, text: str, output_dir: Path = LIVE_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _safe_filename(source.source_id)
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(
        "\n".join(
            [
                f"<!-- source_id: {source.source_id} -->",
                f"<!-- source_url: {source.url} -->",
                f"<!-- source_title: {source.title} -->",
                f"<!-- fetched_at: {fetched_at} -->",
                "",
                f"# {source.title}",
                "",
                text,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def refresh_sources(
    sources: list[SourceDefinition] | None = None,
    output_dir: Path = LIVE_DIR,
    timeout: int = 20,
) -> list[Path]:
    sources = sources or load_source_definitions()
    written: list[Path] = []
    failures: list[str] = []
    for source in sources:
        try:
            text = fetch_source(source, timeout=timeout)
            written.append(write_cached_source(source, text, output_dir))
            print(f"Fetched {source.source_id}: {len(text):,} characters")
        except Exception as exc:  # noqa: BLE001 - report each source and finish the batch
            failures.append(f"{source.source_id}: {exc}")
            print(f"Fetch failed for {source.source_id}: {exc}")
    if failures:
        raise RuntimeError("Live source refresh failed:\n" + "\n".join(failures))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the live Nationalpark Schwarzwald visitor-information corpus.")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()
    paths = refresh_sources(timeout=args.timeout)
    print(f"Fetched {len(paths)} live sources into {LIVE_DIR}")


if __name__ == "__main__":
    main()
