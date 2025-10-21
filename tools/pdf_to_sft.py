#!/usr/bin/env python3
"""Convert PDF research papers into supervised fine-tuning data.

This utility reads one or more PDF files, extracts their text, splits the
content into manageable chunks, and emits a JSON Lines file that can be used
with ``train_sft.py``. Each output record contains a ``prompt`` that instructs
the model what to do with the text and a ``code`` field that serves as the
training target.

Examples
--------
Create a dataset from all PDFs in ``papers/`` and save it as
``data/papers.jsonl``::

    python tools/pdf_to_sft.py \
        --input papers \
        --output data/papers.jsonl \
        --chunk-size 800 \
        --chunk-overlap 120

By default, the generated prompt asks the model to总结 the provided chunk and
the target text is simply the original chunk. You can customise both pieces via
``--instruction-template`` and ``--target-template``.

Requirements
------------
This script depends on :mod:`pypdf` (preferred) or :mod:`PyPDF2`. Install one of
these libraries before running the script::

    pip install pypdf

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from a PDF file.

    The function tries to import :mod:`pypdf` first and falls back to
    :mod:`PyPDF2` if necessary. An informative error message is raised when no
    suitable backend is available.
    """

    try:  # Prefer the actively maintained package name.
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError(
                "Missing dependency: install either 'pypdf' or 'PyPDF2' to read PDF files."
            ) from exc

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned = text.strip()
        if cleaned:
            pages.append(cleaned)
    return "\n\n".join(pages)


def iter_pdf_files(paths: Sequence[Path]) -> Iterator[Path]:
    """Yield all PDF files under the provided paths."""

    for path in paths:
        if path.is_dir():
            for pdf_path in sorted(path.rglob("*.pdf")):
                if pdf_path.is_file():
                    yield pdf_path
        elif path.is_file() and path.suffix.lower() == ".pdf":
            yield path


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split *text* into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    text = " ".join(text.split())  # normalise whitespace
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def build_records(
    pdf_paths: Iterable[Path],
    chunk_size: int,
    chunk_overlap: int,
    instruction_template: str,
    target_template: str,
) -> Iterator[dict]:
    """Create JSONL records for all PDF chunks."""

    for pdf_path in pdf_paths:
        text = extract_text_from_pdf(pdf_path)
        for index, chunk in enumerate(chunk_text(text, chunk_size, chunk_overlap), start=1):
            prompt = instruction_template.format(chunk=chunk, source=str(pdf_path), index=index)
            target = target_template.format(chunk=chunk, source=str(pdf_path), index=index)
            yield {
                "prompt": prompt,
                "code": target,
                "metadata": {
                    "source": str(pdf_path),
                    "chunk_index": index,
                },
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF papers to JSONL for SFT training")
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One or more PDF files or directories containing PDFs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Maximum number of characters per chunk (default: 1024).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Number of characters to overlap between consecutive chunks (default: 128).",
    )
    parser.add_argument(
        "--instruction-template",
        type=str,
        default="请阅读以下论文片段并总结其关键技术要点：\n\n{chunk}",
        help="Template used to build the prompt. Available variables: {chunk}, {source}, {index}.",
    )
    parser.add_argument(
        "--target-template",
        type=str,
        default="{chunk}",
        help="Template used to build the target output. Variables: {chunk}, {source}, {index}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = [Path(path).expanduser() for path in args.input]
    pdf_paths = list(iter_pdf_files(input_paths))
    if not pdf_paths:
        raise FileNotFoundError("No PDF files found in the provided input paths.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = build_records(
        pdf_paths=pdf_paths,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        instruction_template=args.instruction_template,
        target_template=args.target_template,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote dataset with PDFs from {len(pdf_paths)} files to {output_path}")


if __name__ == "__main__":
    main()
