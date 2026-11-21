"""Build a permissively licensed code corpus for the Stage 2.1 tokenizer candidate.

The Stage 2 corpus contained 14.9 MiB of code (1.8%). This script adds code from
public repositories whose licences allow redistribution of derived text, records
provenance per source, and writes train/held-out shards in the same format as
``prepare_corpus.py`` so the trainer can consume them unchanged.

Rules applied to every file:
  * UTF-8 only (files that fail strict decoding are skipped, so no binaries);
  * CRLF -> LF, NFC, no NUL/C0 control characters;
  * skip minified bundles, source maps, lock files and vendored trees;
  * skip files < 200 bytes or > 1 MiB, and files with > 10% non-ASCII;
  * exact-duplicate documents removed by SHA-256;
  * deterministic split: hash % 20 == 0 goes to held-out (5%).

Usage:
    python3 tokenizer_training/prepare_code_corpus.py \
        --out-dir /workspace/corpus/code \
        --manifest tokenizer_training/manifests/code_corpus_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

CODE_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "documentation",
    ".rst": "documentation",
}

SKIP_DIR_PARTS = {
    "node_modules", "vendor", "third_party", "third-party", "dist", "build",
    ".git", ".github", "coverage", "bower_components", "__pycache__", "site-packages",
    "fixtures", "snapshots", "testdata", "test-data", "generated", "gen",
}
SKIP_SUFFIXES = (
    ".min.js", ".min.css", ".map", "-lock.json", ".lock", ".pb.go", ".g.dart",
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf",
)
SKIP_NAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
    "poetry.lock", "Cargo.lock", "Gemfile.lock",
}

# repo, ref, SPDX licence, licence file, byte budget (MiB), extensions to keep
SOURCES: list[dict] = [
    {"repo": "python/cpython", "ref": "main", "license": "PSF-2.0", "license_file": "LICENSE",
     "budget_mib": 6, "exts": {".py"}, "paths": ("Lib/",)},
    {"repo": "python/cpython", "ref": "main", "license": "PSF-2.0", "license_file": "LICENSE",
     "budget_mib": 2, "exts": {".rst"}, "paths": ("Doc/",), "kind": "documentation"},
    {"repo": "TheAlgorithms/Python", "ref": "master", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 3, "exts": {".py"}, "paths": None},
    {"repo": "pallets/flask", "ref": "main", "license": "BSD-3-Clause", "license_file": "LICENSE.txt",
     "budget_mib": 2, "exts": {".py"}, "paths": ("src/", "tests/")},
    {"repo": "psf/requests", "ref": "main", "license": "Apache-2.0", "license_file": "LICENSE",
     "budget_mib": 2, "exts": {".py"}, "paths": ("src/", "tests/")},
    {"repo": "google/guava", "ref": "master", "license": "Apache-2.0", "license_file": "LICENSE",
     "budget_mib": 5, "exts": {".java"}, "paths": ("guava/src/",)},
    {"repo": "apache/commons-lang", "ref": "master", "license": "Apache-2.0", "license_file": "LICENSE.txt",
     "budget_mib": 3, "exts": {".java"}, "paths": ("src/main/java/",)},
    {"repo": "TheAlgorithms/Java", "ref": "master", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 4, "exts": {".java"}, "paths": None},
    {"repo": "microsoft/TypeScript", "ref": "main", "license": "Apache-2.0", "license_file": "LICENSE.txt",
     "budget_mib": 6, "exts": {".ts", ".js"}, "paths": ("packages/typescript/src/",)},
    {"repo": "facebook/react", "ref": "main", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 4, "exts": {".js", ".ts"}, "paths": ("packages/",)},
    {"repo": "lodash/lodash", "ref": "main", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 2, "exts": {".js"}, "paths": None},
    {"repo": "axios/axios", "ref": "v1.x", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 2, "exts": {".js", ".ts"}, "paths": ("lib/", "test/")},
    {"repo": "postgres/postgres", "ref": "master", "license": "PostgreSQL", "license_file": "COPYRIGHT",
     "budget_mib": 6, "exts": {".sql"}, "paths": ("src/test/", "src/backend/catalog/")},
    {"repo": "twbs/bootstrap", "ref": "main", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 5, "exts": {".css", ".scss", ".html"}, "paths": ("scss/", "site/")},
    {"repo": "tailwindlabs/tailwindcss", "ref": "main", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 3, "exts": {".css", ".html", ".ts"}, "paths": ("packages/",)},
    {"repo": "vuejs/core", "ref": "main", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 2, "exts": {".ts", ".js"}, "paths": ("packages/",)},
    {"repo": "prettier/prettier", "ref": "main", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 2, "exts": {".js", ".ts"}, "paths": ("src/",)},
    {"repo": "expressjs/express", "ref": "master", "license": "MIT", "license_file": "LICENSE",
     "budget_mib": 1, "exts": {".js"}, "paths": ("lib/",)},
    {"repo": "apache/commons-io", "ref": "master", "license": "Apache-2.0", "license_file": "LICENSE.txt",
     "budget_mib": 2, "exts": {".java"}, "paths": ("src/main/java/",)},
    {"repo": "apache/calcite", "ref": "main", "license": "Apache-2.0", "license_file": "LICENSE",
     "budget_mib": 4, "exts": {".java", ".sql"}, "paths": None},
    {"repo": "apache/spark", "ref": "master", "license": "Apache-2.0", "license_file": "LICENSE",
     "budget_mib": 4, "exts": {".java", ".sql"}, "paths": None},
    {"repo": "pandas-dev/pandas", "ref": "main", "license": "BSD-3-Clause", "license_file": "LICENSE",
     "budget_mib": 3, "exts": {".py"}, "paths": ("pandas/",)},
    {"repo": "scikit-learn/scikit-learn", "ref": "main", "license": "BSD-3-Clause", "license_file": "COPYING",
     "budget_mib": 3, "exts": {".py"}, "paths": ("sklearn/",)},
    {"repo": "django/django", "ref": "main", "license": "BSD-3-Clause", "license_file": "LICENSE",
     "budget_mib": 3, "exts": {".rst", ".txt"}, "paths": ("docs/",), "kind": "documentation"},
    {"repo": "rust-lang/book", "ref": "main", "license": "MIT", "license_file": "LICENSE-MIT",
     "budget_mib": 2, "exts": {".md"}, "paths": ("src/",), "kind": "documentation"},
]

CONFIG_EXTS = {".json", ".yaml", ".yml"}
CONFIG_BUDGET_MIB = 4  # gathered opportunistically from all sources above


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 0x20)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def keep_file(name: str, exts: set[str]) -> bool:
    parts = name.split("/")
    if any(part in SKIP_DIR_PARTS for part in parts[:-1]):
        return False
    if Path(name).name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
        return False
    suffix = Path(name).suffix.lower()
    return suffix in exts


def fetch(repo: str, ref: str) -> bytes:
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{ref}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            if attempt == 2:
                raise SystemExit(f"download failed for {repo}@{ref}: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="/workspace/corpus/code")
    parser.add_argument("--manifest", default="tokenizer_training/manifests/code_corpus_manifest.json")
    parser.add_argument("--max-file-bytes", type=int, default=1_000_000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    eval_dir = out_dir / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    documents: list[tuple[str, str, str]] = []  # (kind, source, text)
    per_source: dict[str, dict] = {}
    licences: dict[str, dict] = {}
    config_bytes = 0

    for spec in SOURCES:
        repo, ref = spec["repo"], spec["ref"]
        key = f"{repo}@{ref}"
        exts = spec["exts"]
        budget = spec["budget_mib"] * 2**20
        paths = spec.get("paths")
        kind = spec.get("kind", "code")
        print(f"[fetch] {key} ({spec['license']}, budget {spec['budget_mib']} MiB)", flush=True)
        blob = fetch(repo, ref)
        used = 0
        kept = 0
        licence_text = None

        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = member.name.split("/", 1)[1] if "/" in member.name else member.name
                if Path(name).name == spec["license_file"] and licence_text is None:
                    licence_text = tar.extractfile(member).read(4096).decode("utf-8", "replace")
                if member.size < 200 or member.size > args.max_file_bytes:
                    continue
                if paths and not any(name.startswith(p) for p in paths):
                    continue
                suffix = Path(name).suffix.lower()
                is_config = suffix in CONFIG_EXTS
                if not is_config and not keep_file(name, exts):
                    continue
                if is_config and config_bytes >= CONFIG_BUDGET_MIB * 2**20:
                    continue
                if used >= budget and not is_config:
                    continue
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                try:
                    raw = handle.read().decode("utf-8")
                except UnicodeDecodeError:
                    continue
                text = normalise(raw)
                if len(text.encode("utf-8")) < 200:
                    continue
                non_ascii = sum(1 for ch in text if ord(ch) > 127)
                if non_ascii > 0.10 * len(text):
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                documents.append((kind, key, text))
                kept += 1
                size = len(text.encode("utf-8"))
                used += size
                if is_config:
                    config_bytes += size

        licences[key] = {
            "repo": repo,
            "ref": ref,
            "license": spec["license"],
            "license_file": spec["license_file"],
            "license_header": (licence_text or "").splitlines()[:3],
        }
        per_source[key] = {
            "license": spec["license"],
            "kind": kind,
            "files": kept,
            "bytes": used,
            "budget_mib": spec["budget_mib"],
        }
        print(f"[fetch] {key}: {kept} files, {used / 2**20:.2f} MiB", flush=True)

    train_docs = [d for d in documents if int(hashlib.sha256(d[2].encode()).hexdigest()[:8], 16) % 20 != 0]
    eval_docs = [d for d in documents if int(hashlib.sha256(d[2].encode()).hexdigest()[:8], 16) % 20 == 0]

    def write(docs, directory: Path) -> tuple[int, int]:
        shards, buffer, size, total = 0, [], 0, 0
        for kind, source, text in docs:
            block = f"<doc kind=\"{kind}\" source=\"{source}\">\n{text}\n</doc>\n\n"
            buffer.append(block)
            size += len(block)
            if size >= 32 * 2**20:
                (directory / f"code-{shards:04d}.txt").write_text("".join(buffer), encoding="utf-8")
                shards += 1
                total += size
                buffer, size = [], 0
        if buffer:
            (directory / f"code-{shards:04d}.txt").write_text("".join(buffer), encoding="utf-8")
            shards += 1
            total += size
        return shards, total

    train_shards, train_bytes = write(train_docs, train_dir)
    eval_shards, eval_bytes = write(eval_docs, eval_dir)

    by_kind: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)
    for kind, source, text in documents:
        size = len(text.encode("utf-8"))
        by_kind[kind] += size
        by_language[source] += size

    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "code-enriched corpus for the Stage 2.1 tokenizer candidate (not for language-model pretraining)",
        "rules": {
            "encoding": "utf-8 strict",
            "normalisation": "NFC, CRLF->LF, C0 controls removed, 3+ newlines collapsed",
            "min_file_bytes": 200,
            "max_file_bytes": args.max_file_bytes,
            "max_non_ascii_fraction": 0.10,
            "dedupe": "sha256 of normalised text",
            "heldout": "sha256(text)[:8] % 20 == 0 (5%)",
            "skipped_dirs": sorted(SKIP_DIR_PARTS),
            "skipped_names": sorted(SKIP_NAMES),
        },
        "sources": licences,
        "statistics": {
            "documents": len(documents),
            "bytes": sum(len(t.encode()) for _, _, t in documents),
            "train_documents": len(train_docs),
            "train_bytes": train_bytes,
            "train_shards": train_shards,
            "eval_documents": len(eval_docs),
            "eval_bytes": eval_bytes,
            "eval_shards": eval_shards,
            "config_bytes": config_bytes,
            "per_source": per_source,
            "per_kind": dict(by_kind),
            "per_language": dict(by_language),
        },
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "train_mib": round(train_bytes / 2**20, 2),
        "eval_mib": round(eval_bytes / 2**20, 2),
        "documents": len(documents),
        "per_kind_mib": {k: round(v / 2**20, 2) for k, v in by_kind.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
