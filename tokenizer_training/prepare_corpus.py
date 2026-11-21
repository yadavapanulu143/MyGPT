"""Build the tokenizer-training corpus with provenance and licences.

Sources (all English, all redistributable for this purpose):

  wikitext-103-raw   CC BY-SA 3.0        encyclopedic prose (Wikipedia)
  project-gutenberg  Public domain       books, dialogue, older prose
  oasst1             Apache-2.0          conversational English
  python-stdlib      PSF-2.0             code, identifiers, punctuation
  ietf-rfc           IETF Trust (free)   technical text, URLs, numbers, abbreviations

The corpus is deliberately small and reproducible. It is *not* the corpus for
language-model pretraining: tokenizer training only needs enough text to settle
a 32k merge list, and keeping the two separate avoids leaking eval text into the
pretraining set later.

Processing is deterministic: sources are read in a fixed order, documents keep
their file order, and no sampling or shuffling is applied.

Usage:
    python3 tokenizer_training/prepare_corpus.py --out-dir /workspace/corpus/processed
    python3 tokenizer_training/prepare_corpus.py --limit-train-mb 25 --out-dir /workspace/corpus/pilot25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets"
RFC_BASE = "https://www.rfc-editor.org/rfc"

# A small, well-known slice of RFCs: HTTP, TCP/IP, mail, TLS, URIs, JSON, CSV.
RFC_IDS = [
    791, 793, 821, 822, 959, 1034, 1035, 1122, 1123, 1738, 1939, 2045, 2046,
    2131, 2324, 2616, 2821, 2822, 3339, 3986, 3987, 4180, 5234, 5321, 5322,
    5890, 6265, 6455, 7230, 7231, 7232, 7233, 7234, 7235, 7519, 7540, 7541,
    8259, 8446, 9110, 9111, 9112, 9113, 9114,
]


@dataclass
class Source:
    name: str
    kind: str  # "parquet" | "local" | "rfc"
    license: str
    license_url: str
    urls: list[str] = field(default_factory=list)
    path_glob: str | None = None
    text_field: str = "text"
    language_filter: bool = False


SOURCES = [
    Source(
        name="wikitext-103-raw",
        kind="parquet",
        license="CC BY-SA 3.0",
        license_url="https://creativecommons.org/licenses/by-sa/3.0/",
        urls=[
            f"{HF_BASE}/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet",
            f"{HF_BASE}/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet",
        ],
    ),
    Source(
        name="project-gutenberg",
        kind="parquet",
        license="Public domain (Project Gutenberg)",
        license_url="https://www.gutenberg.org/policy/license.html",
        urls=[
            f"{HF_BASE}/manu/project_gutenberg/resolve/main/data/en-00000-of-00052-7cda8f63c262acf8.parquet",
        ],
    ),
    Source(
        name="oasst1",
        kind="parquet",
        license="Apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        urls=[
            f"{HF_BASE}/OpenAssistant/oasst1/resolve/main/data/train-00000-of-00001-b42a775f407cee45.parquet",
        ],
        language_filter=True,
    ),
    Source(
        name="python-stdlib",
        kind="local",
        license="PSF-2.0",
        license_url="https://docs.python.org/3/license.html",
        path_glob="/usr/local/lib/python3.*/**/*.py",
    ),
    Source(
        name="ietf-rfc",
        kind="rfc",
        license="IETF Trust Legal Provisions (free to reproduce)",
        license_url="https://trustee.ietf.org/license-info/",
        urls=[f"{RFC_BASE}/rfc{r}.txt" for r in RFC_IDS],
    ),
]

_NEWLINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize(text: str) -> str:
    """Conservative normalisation: NFC, LF newlines, no control junk.

    Deliberately not NFKC and no case folding: the tokenizer should see the
    characters a user actually types, including punctuation and spacing.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)
    text = _TRAILING_SPACE.sub("\n", text)
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()


def is_english(text: str, max_non_ascii: float = 0.05) -> bool:
    if not text:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text) <= max_non_ascii


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as response, open(tmp, "wb") as fh:
                while chunk := response.read(1 << 20):
                    fh.write(chunk)
            tmp.rename(dest)
            return dest
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"  download failed ({attempt + 1}/3): {exc}", file=sys.stderr)
            time.sleep(3)
    raise RuntimeError(f"could not download {url}")


def iter_parquet_documents(path: Path, text_field: str, language_filter: bool, chunk_chars: int = 2000):
    """Yield documents, merging consecutive rows into ~chunk_chars blocks.

    wikitext and oasst store one line or one utterance per row; treating each row
    as a document would make dedupe and the eval split meaningless, and would
    discard most rows as "too short".
    """
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    columns = [c for c in (text_field, "lang") if c in parquet.schema_arrow.names]
    buffer: list[str] = []
    size = 0
    for batch in parquet.iter_batches(batch_size=2048, columns=columns):
        data = batch.to_pydict()
        texts = data.get(text_field) or []
        langs = data.get("lang") or [None] * len(texts)
        for text, lang in zip(texts, langs):
            if not text:
                continue
            if language_filter and lang not in (None, "en", "en-US", "en-GB"):
                continue
            buffer.append(text)
            size += len(text)
            if size >= chunk_chars:
                yield "\n".join(buffer)
                buffer, size = [], 0
    if buffer:
        yield "\n".join(buffer)


def iter_local_documents(pattern: str):
    import glob

    for name in sorted(glob.glob(pattern, recursive=True)):
        if "/test/" in name or name.endswith("_test.py"):
            continue
        try:
            yield Path(name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def iter_rfc_documents(raw_dir: Path, urls: list[str]):
    for url in urls:
        name = url.rsplit("/", 1)[-1]
        path = download(url, raw_dir / "rfc" / name)
        yield path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="/workspace/corpus/raw")
    parser.add_argument("--out-dir", default="/workspace/corpus/processed")
    parser.add_argument("--manifest", default="tokenizer_training/manifests/corpus_manifest.json")
    parser.add_argument("--limit-train-mb", type=float, default=None, help="stop after this much train text")
    parser.add_argument("--limit-source-mb", type=float, default=None, help="cap train bytes per source (pilot runs)")
    parser.add_argument("--shard-mb", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=100, help="1 document in N goes to the eval split")
    parser.add_argument("--chunk-chars", type=int, default=2000, help="merge parquet rows into documents of this size")
    parser.add_argument("--sources", default=None, help="comma separated subset of source names")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.sources.split(",")) if args.sources else None
    limit_bytes = int(args.limit_train_mb * 2**20) if args.limit_train_mb else None
    source_limit = int(args.limit_source_mb * 2**20) if args.limit_source_mb else None
    shard_bytes = int(args.shard_mb * 2**20)

    seen: set[str] = set()
    stats: dict[str, dict] = defaultdict(lambda: {"documents": 0, "kept": 0, "bytes": 0, "duplicates": 0, "filtered": 0})

    shard_index = -1
    shard_fh = None
    shard_size = 0
    shard_hashes: list[dict] = []
    train_bytes = 0
    eval_fh = open(eval_dir / "heldout.txt", "w", encoding="utf-8", newline="\n")
    eval_bytes = 0
    eval_docs = 0
    global_doc = 0
    stop = False

    def close_shard():
        nonlocal shard_fh, shard_size
        if shard_fh is None:
            return
        shard_fh.close()
        shard_fh = None
        digest = hashlib.sha256()
        path = train_dir / f"train-{shard_index:04d}.txt"
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        shard_hashes.append({"file": path.name, "sha256": digest.hexdigest(), "bytes": shard_size})
        shard_size = 0

    def open_shard():
        nonlocal shard_fh, shard_index
        close_shard()
        shard_index += 1
        shard_fh = open(train_dir / f"train-{shard_index:04d}.txt", "w", encoding="utf-8", newline="\n")

    for source in SOURCES:
        if wanted and source.name not in wanted:
            continue
        print(f"[source] {source.name} ({source.license})", flush=True)
        if source.kind == "parquet":
            paths = [download(url, raw_dir / source.name / url.rsplit("/", 1)[-1]) for url in source.urls]
            documents = (
                doc
                for path in paths
                for doc in iter_parquet_documents(path, source.text_field, source.language_filter, args.chunk_chars)
            )
        elif source.kind == "local":
            documents = iter_local_documents(source.path_glob or "")
        else:
            documents = iter_rfc_documents(raw_dir, source.urls)

        for raw_text in documents:
            stats[source.name]["documents"] += 1
            text = normalize(raw_text)
            if len(text) < 100 or not is_english(text):
                stats[source.name]["filtered"] += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                stats[source.name]["duplicates"] += 1
                continue
            seen.add(digest)

            payload = text + "\n\n"
            size = len(payload.encode("utf-8"))
            stats[source.name]["kept"] += 1
            stats[source.name]["bytes"] += size
            global_doc += 1

            if source_limit and stats[source.name]["bytes"] >= source_limit:
                stats[source.name]["capped"] = True
                break
            if global_doc % args.eval_every == 0:
                eval_fh.write(payload)
                eval_bytes += size
                eval_docs += 1
            else:
                if shard_fh is None:
                    open_shard()
                shard_fh.write(payload)
                shard_size += size
                train_bytes += size
                if shard_size >= shard_bytes:
                    close_shard()
                if limit_bytes and train_bytes >= limit_bytes:
                    stop = True
                    break
        if stop:
            break

    eval_fh.close()
    close_shard()

    manifest = {
        "purpose": "tokenizer training corpus (not language-model pretraining data)",
        "generated_by": "tokenizer_training/prepare_corpus.py",
        "normalization": "unicode NFC; CRLF->LF; C0 controls removed; trailing spaces stripped; 3+ newlines -> 2",
        "deduplication": "exact sha256 of normalised document text",
        "filters": {"min_document_chars": 100, "max_non_ascii_ratio": 0.05},
        "split": {"eval_every_n_documents": args.eval_every},
        "parquet_chunk_chars": args.chunk_chars,
        "limit_train_mb": args.limit_train_mb,
        "limit_source_mb": args.limit_source_mb,
        "sources": [
            {
                "name": s.name,
                "license": s.license,
                "license_url": s.license_url,
                "urls": s.urls,
                "path_glob": s.path_glob,
            }
            for s in SOURCES
            if not wanted or s.name in wanted
        ],
        "statistics": {
            "per_source": stats,
            "train_bytes": train_bytes,
            "train_documents": global_doc - eval_docs,
            "eval_bytes": eval_bytes,
            "eval_documents": eval_docs,
            "duplicates_dropped": sum(v["duplicates"] for v in stats.values()),
            "filtered_dropped": sum(v["filtered"] for v in stats.values()),
        },
        "train_shards": shard_hashes,
        "eval_file": {"file": "eval/heldout.txt", "bytes": eval_bytes},
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["statistics"], indent=2))
    print(f"train: {train_bytes / 2**20:.1f} MiB in {len(shard_hashes)} shard(s)")
    print(f"eval : {eval_bytes / 2**20:.2f} MiB, {eval_docs} documents")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
