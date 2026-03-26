from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


DATASETS_DIR = Path("datasets")


@dataclass(frozen=True)
class CorpusSpec:
    dataset_id: str
    display_name: str
    local_path: str
    url: str | None = None
    start_marker: str | None = None
    end_marker: str | None = None
    char_limit: int | None = None


def normalize_text(text: str) -> str:
    text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def strip_between_markers(text: str, start_marker: str | None, end_marker: str | None) -> str:
    text = text.lstrip("\ufeff")
    if start_marker and start_marker in text:
        text = text.split(start_marker, 1)[1]
    elif "*** START OF THE PROJECT GUTENBERG EBOOK" in text:
        parts = re.split(
            r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*",
            text,
            maxsplit=1,
            flags=re.S,
        )
        text = parts[-1]
    if end_marker and end_marker in text:
        text = text.split(end_marker, 1)[0]
    elif "*** END OF THE PROJECT GUTENBERG EBOOK" in text:
        parts = re.split(
            r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .*? \*\*\*",
            text,
            maxsplit=1,
            flags=re.S,
        )
        text = parts[0]
    return text


def download_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def ensure_corpus(spec: CorpusSpec) -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    direct_path = Path(spec.local_path)
    if direct_path.exists():
        return direct_path
    output_path = DATASETS_DIR / spec.local_path
    if output_path.exists():
        return output_path

    if spec.url is None:
        raise FileNotFoundError(f"Local corpus not found: {output_path}")

    raw_text = download_text(spec.url)
    text = strip_between_markers(raw_text, spec.start_marker, spec.end_marker)
    text = normalize_text(text)
    if spec.char_limit is not None:
        text = text[: spec.char_limit].rstrip() + "\n"
    output_path.write_text(text, encoding="utf-8")
    return output_path


def write_dataset_manifest(output_path: Path, specs: list[CorpusSpec]) -> None:
    payload = [asdict(spec) for spec in specs]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
