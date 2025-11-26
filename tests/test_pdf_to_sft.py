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
"""Tests for the pdf_to_sft module."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools import pdf_to_sft


def test_iter_pdf_files(tmp_path: Path) -> None:
    """Test the iter_pdf_files function."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf1 = pdf_dir / "paper1.pdf"
    pdf1.touch()
    pdf2 = pdf_dir / "paper2.pdf"
    pdf2.touch()
    txt_file = pdf_dir / "notes.txt"
    txt_file.touch()

    pdf_files = list(pdf_to_sft.iter_pdf_files([pdf_dir]))
    assert len(pdf_files) == 2
    assert pdf1 in pdf_files
    assert pdf2 in pdf_files


def test_chunk_text() -> None:
    """Test the chunk_text function."""
    text = "This is a test sentence. This is another test sentence."
    chunks = pdf_to_sft.chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 4
    assert chunks[0][0] == "This is a test sente"
    assert chunks[1][0] == "sentence. This is an"
    assert chunks[2][0] == "is another test sent"
    assert chunks[3][0] == " sentence."


def test_strip_reference_section() -> None:
    """Test the _strip_reference_section function."""
    text = "This is the main content.\n\nREFERENCES\nThis is a reference."
    stripped_text = pdf_to_sft._strip_reference_section(text)
    assert stripped_text == "This is the main content."

    text = "This is the main content.\n\nBIBLIOGRAPHY\nThis is a reference."
    stripped_text = pdf_to_sft._strip_reference_section(text)
    assert stripped_text == "This is the main content."


def test_clean_page_text() -> None:
    """Test the _clean_page_text function."""
    text = "This is a    test sentence.\n\nThis is another test sen-\ntence."
    cleaned_text = pdf_to_sft._clean_page_text(text)
    assert cleaned_text == "This is a test sentence.\n\nThis is another test sentence."
