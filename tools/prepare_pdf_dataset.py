#!/usr/bin/env python3
# Copyright 2024 The Qwen2.5-Coder-7B-Instruct Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build training/validation datasets from a folder of quantum PDFs."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from tools import pdf_to_sft
else:
    from . import pdf_to_sft

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


def _write_jsonl(records: Sequence[dict], path: Path) -> None:
    """Write *records* to *path* in JSON Lines format."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")


def _collect_records(
    pdf_paths: Sequence[Path],
    *,
    chunk_size: int,
    chunk_overlap: int,
    instruction_template: str,
    target_template: str,
    analysis_template: Optional[str],
    min_chunk_length: int,
    strip_references: bool,
    dedupe: bool,
) -> List[dict]:
    """Collect JSONL records from *pdf_paths*."""
    records: List[Dict] = []
    seen_targets: set[str] = set()

    if tqdm:
        pdf_paths = tqdm(pdf_paths, desc="Processing PDFs")

    for record in pdf_to_sft.build_records(
        pdf_paths=pdf_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        instruction_template=instruction_template,
        target_template=target_template,
        analysis_template=analysis_template,
        min_chunk_length=min_chunk_length,
        strip_references=strip_references,
    ):
        if dedupe:
            target_text = record.get("code") or ""
            if target_text in seen_targets:
                continue
            seen_targets.add(target_text)
        records.append(record)
    return records


def _split_records(records: List[dict], train_ratio: float) -> Tuple[List[dict], List[dict]]:
    """Split *records* into train and validation sets."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive)")

    split_index = max(1, int(len(records) * train_ratio))
    if split_index >= len(records):
        split_index = len(records) - 1
    train_records = records[:split_index]
    valid_records = records[split_index:]
    if not train_records or not valid_records:
        # Fallback: ensure at least one record on each side when possible.
        midpoint = len(records) // 2 or 1
        train_records = records[:midpoint]
        valid_records = records[midpoint:]
    return train_records, valid_records


def prepare_datasets(
    pdf_dir: Path,
    output_dir: Path,
    *,
    dataset_name: Optional[str] = None,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    instruction_template: str = "Please read the following paper snippet and summarize its key technical points:\n\n{chunk}",
    target_template: str = "{chunk}",
    analysis_template: Optional[str] = None,
    min_chunk_length: int = 128,
    strip_references: bool = False,
    dedupe: bool = False,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Process PDFs under *pdf_dir* and save datasets into *output_dir*."""
    if not pdf_dir.exists():
        logger.warning("PDF directory '%s' does not exist; skipping.", pdf_dir)
        return None, None

    pdf_paths = list(pdf_to_sft.iter_pdf_files([pdf_dir]))
    if not pdf_paths:
        logger.warning("No PDF files found under '%s'; skipping.", pdf_dir)
        return None, None

    records = _collect_records(
        pdf_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        instruction_template=instruction_template,
        target_template=target_template,
        analysis_template=analysis_template,
        min_chunk_length=min_chunk_length,
        strip_references=strip_references,
        dedupe=dedupe,
    )

    if not records:
        raise ValueError("No records were generated from the provided PDFs")

    random.Random(seed).shuffle(records)

    if len(records) < 2:
        train_records = records
        valid_records: List[dict] = []
    else:
        train_records, valid_records = _split_records(records, train_ratio)

    prefix = dataset_name or ""
    if prefix:
        train_path = output_dir / f"{prefix}_train.jsonl"
        valid_path = output_dir / f"{prefix}_valid.jsonl"
        all_path = output_dir / f"{prefix}_all.jsonl"
    else:
        train_path = output_dir / "train.jsonl"
        valid_path = output_dir / "valid.jsonl"
        all_path = output_dir / "all.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(records, all_path)
    _write_jsonl(train_records, train_path)
    if valid_records:
        _write_jsonl(valid_records, valid_path)
    else:
        valid_path = None

    logger.info(
        "Prepared %s total records (%s train / %s valid) from %s PDFs",
        len(records),
        len(train_records),
        len(valid_records),
        len(pdf_paths),
    )

    return train_path, valid_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a folder of PDFs into JSONL datasets ready for fine-tuning",
    )
    parser.add_argument(
        "--pdf-dir",
        required=True,
        help="The folder that contains PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="The directory where processed datasets will be stored (default: data/processed).",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="An optional prefix used for the generated dataset files (e.g., quantum_papers).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="The maximum number of characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="The number of characters to overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--instruction-template",
        default="Please read the following paper snippet and summarize its key technical points:\n\n{chunk}",
        help="The template used for the prompt field.",
    )
    parser.add_argument(
        "--target-template",
        default="{chunk}",
        help="The template used for the code/target field.",
    )
    parser.add_argument(
        "--analysis-template",
        default=None,
        help="An optional template for the analysis field.",
    )
    parser.add_argument(
        "--min-chunk-length",
        type=int,
        default=128,
        help="Discard chunks shorter than this length.",
    )
    parser.add_argument(
        "--strip-references",
        action="store_true",
        help="Remove reference sections from PDFs before splitting.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate chunks based on the code field.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="The fraction of samples reserved for training (the rest is used for validation).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed for shuffling.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """The main function for the PDF dataset preparation script."""
    try:
        import pypdf
    except ImportError:
        print("Please install the required dependencies: `pip install pypdf`")
        return

    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s: %(message)s")
    pdf_dir = Path(args.pdf_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    train_path, valid_path = prepare_datasets(
        pdf_dir,
        output_dir,
        dataset_name=args.dataset_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        instruction_template=args.instruction_template,
        target_template=args.target_template,
        analysis_template=args.analysis_template,
        min_chunk_length=args.min_chunk_length,
        strip_references=args.strip_references,
        dedupe=args.dedupe,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    if train_path is None:
        message = f"No PDFs were processed; please check if the directory '{pdf_dir}' exists and contains PDF files."
        logger.warning(message)
        print(message)
        return

    message = [f"Processing complete, data saved to '{output_dir}'", f"Training set = {train_path}"]
    if valid_path:
        message.append(f"Validation set = {valid_path}")
    else:
        message.append("Validation set = <not generated>")
    logger.info("%s", "; ".join(message))
    print("; ".join(message))


if __name__ == "__main__":
    main()
