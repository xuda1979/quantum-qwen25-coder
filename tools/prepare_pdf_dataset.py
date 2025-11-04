#!/usr/bin/env python3
"""Build training/validation datasets from a folder of quantum PDFs."""

from __future__ import annotations

import argparse

import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from tools import pdf_to_sft


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

    records: List[dict] = []
    seen_targets: set[str] = set()
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
    instruction_template: str = "请阅读以下论文片段并总结其关键技术要点：\n\n{chunk}",
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
        raise FileNotFoundError(f"PDF directory '{pdf_dir}' does not exist")

    pdf_paths = list(pdf_to_sft.iter_pdf_files([pdf_dir]))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under '{pdf_dir}'")

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
    parser = argparse.ArgumentParser(
        description="Convert a folder of PDFs into JSONL datasets ready for fine-tuning",
    )
    parser.add_argument("--pdf-dir", required=True, help="Folder that contains PDF files")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory where processed datasets will be stored (default: data/processed)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional prefix used for the generated dataset files (e.g. quantum_papers)",
    )
    parser.add_argument("--chunk-size", type=int, default=1024, help="Maximum characters per chunk")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Number of characters to overlap between consecutive chunks",
    )
    parser.add_argument(
        "--instruction-template",
        default="请阅读以下论文片段并总结其关键技术要点：\n\n{chunk}",
        help="Template used for the prompt field",
    )
    parser.add_argument(
        "--target-template",
        default="{chunk}",
        help="Template used for the code/target field",
    )
    parser.add_argument(
        "--analysis-template",
        default=None,
        help="Optional template for the analysis field",
    )
    parser.add_argument(
        "--min-chunk-length",
        type=int,
        default=128,
        help="Discard chunks shorter than this length",
    )
    parser.add_argument(
        "--strip-references",
        action="store_true",
        help="Remove reference sections from PDFs before splitting",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate chunks based on the code field",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Fraction of samples reserved for training (rest used for validation)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
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

    message = [f"处理完成，数据已保存至 '{output_dir}'", f"训练集={train_path}"]
    if valid_path:
        message.append(f"验证集={valid_path}")
    else:
        message.append("验证集=<未生成>")
    logger.info("%s", "; ".join(message))
    print("; ".join(message))


if __name__ == "__main__":
    main()
