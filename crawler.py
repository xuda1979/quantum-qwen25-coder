# Copyright 2024 The Qwen2.5-Coder-7B-Instruct Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Advanced data collection script for quantum programming.

This script is designed to automate the collection of high-quality quantum
programming training corpora, with the following main features:

*   Supports extracting titles, contexts, and code examples from online
    tutorial pages such as the **Qiskit Textbook**.
*   Can download sample programs from selected GitHub repositories (e.g.,
    Qiskit, Cirq, PennyLane) and automatically generate appropriate prompts.
*   Uses the StackExchange API to retrieve popular quantum computing Q&A,
    pairing question bodies with code blocks from the answers.
*   Provides basic quality filtering (code line count, keyword matching) and
    a deduplication mechanism.
*   Allows customization of the output path, minimum code lines, and enabled
    data sources through command-line arguments.

The collected data is written to disk in JSON Lines format, with each record
containing `prompt`, `code`, and `metadata` fields, which can be directly used
for supervised fine-tuning or quality assessment. Please ensure that you
comply with the terms of use of the target sites and control the request
frequency appropriately when crawling in bulk.
"""

import argparse
import html
import json
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Dict, Iterable, List, Optional, Sequence

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

STACKEXCHANGE_API = "https://api.stackexchange.com/2.3"

DEFAULT_QISKIT_TUTORIALS: Sequence[str] = (
    "https://qiskit.org/textbook/ch-ex/foundation/quantum-circuit.html",
    "https://qiskit.org/textbook/ch-algorithms/grover-search.html",
    "https://qiskit.org/textbook/ch-applications/vqe-molecules.html",
)

DEFAULT_GITHUB_FILES: Sequence[Dict[str, Optional[str]]] = (
    {
        "url": "https://raw.githubusercontent.com/qiskit-community/ibm-quantum-challenge-2021/main/lab1/lab1.py",
        "prompt": "Use Qiskit to build and measure a Bell state, understanding the basic operations of quantum entanglement.",
    },
    {
        "url": "https://raw.githubusercontent.com/quantumlib/Cirq/master/examples/bell_inequality.py",
        "prompt": "Use Cirq to simulate a Bell inequality experiment, demonstrating how to perform parameter sweeps and analyze measurement results.",
    },
    {
        "url": "https://raw.githubusercontent.com/PennyLaneAI/pennylane/master/examples/qml_strongly_entangling_layers.py",
        "prompt": "Demonstrate how to build strongly entangling layers and train a variational quantum circuit in PennyLane.",
    },
)

DEFAULT_STACKEXCHANGE_TAGS: Sequence[str] = ("quantum-computing", "qiskit")

KEYWORD_HINTS = ("qiskit", "quantumcircuit", "cirq", "pennylane", "qsharp", "ansatz")


@dataclass
class Record:
    """A single quantum programming data record."""

    prompt: str
    code: str
    metadata: Dict[str, str] = field(default_factory=dict)


class QuantumDataCollector:
    """Encapsulates the collection logic for various data sources."""

    def __init__(
        self,
        *,
        min_code_lines: int = 6,
        keywords: Sequence[str] = KEYWORD_HINTS,
        request_interval: float = 0.8,
    ) -> None:
        self.min_code_lines = min_code_lines
        self.keywords = tuple(keyword.lower() for keyword in keywords)
        self.request_interval = request_interval
        self.records: List[Record] = []
        self._seen_hashes: set[str] = set()

    # ------------------------------------------------------------------
    # Generic utilities
    # ------------------------------------------------------------------
    def _fetch(
        self,
        url: str,
        *,
        params: Optional[Dict[str, str]] = None,
        is_json: bool = False,
    ) -> str | Dict:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=45)
        resp.raise_for_status()
        return resp.json() if is_json else resp.text

    def _add_record(self, prompt: str, code: str, metadata: Dict[str, str]) -> bool:
        prompt = textwrap.dedent(prompt).strip()
        code = textwrap.dedent(code).strip()
        if not prompt or not code:
            return False

        code_lines = [line for line in code.splitlines() if line.strip()]
        if len(code_lines) < self.min_code_lines:
            return False

        lowered = code.lower()
        if self.keywords and not any(keyword in lowered for keyword in self.keywords):
            return False

        digest = sha1(f"{prompt}\n{code}".encode("utf-8")).hexdigest()
        if digest in self._seen_hashes:
            return False

        self._seen_hashes.add(digest)
        self.records.append(Record(prompt=prompt, code=code, metadata=metadata))
        return True

    @staticmethod
    def _clean_html_text(html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text(" ", strip=True)
        return html.unescape(re.sub(r"\s+", " ", text))

    # ------------------------------------------------------------------
    # Qiskit Textbook
    # ------------------------------------------------------------------
    def collect_qiskit_textbook(self, urls: Iterable[str]) -> None:
        """Collect data from the Qiskit Textbook."""
        for url in urls:
            try:
                print(f"[Qiskit] Fetching {url}")
                html_text = self._fetch(url)
                soup = BeautifulSoup(html_text, "html.parser")
                for section in soup.find_all("section"):
                    title_tag = section.find(["h1", "h2", "h3"])
                    if not title_tag:
                        continue
                    description_parts = [
                        p.get_text(" ", strip=True)
                        for p in section.find_all("p", limit=2)
                        if p.get_text(strip=True)
                    ]
                    prompt_prefix = "\n".join(description_parts)
                    prompt = f"{title_tag.get_text(strip=True)}\n\n{prompt_prefix}".strip()
                    for idx, pre_tag in enumerate(section.find_all("pre")):
                        code = pre_tag.get_text()
                        metadata = {
                            "source": "qiskit_textbook",
                            "url": url,
                            "section": title_tag.get_text(strip=True),
                            "snippet_index": str(idx),
                        }
                        self._add_record(prompt, code, metadata)
                time.sleep(self.request_interval)
            except Exception as exc:  # noqa: BLE001
                print(f"[Qiskit] Failed to collect from {url}: {exc}")

    # ------------------------------------------------------------------
    # GitHub raw files
    # ------------------------------------------------------------------
    def collect_github_files(self, resources: Iterable[Dict[str, Optional[str]]]) -> None:
        """Collect data from GitHub raw files."""
        for resource in resources:
            url = resource["url"]
            fallback_prompt = resource.get("prompt")
            try:
                print(f"[GitHub] Fetching {url}")
                code_text = self._fetch(url)
                prompt = fallback_prompt or self._derive_prompt_from_code(code_text, url)
                metadata = {
                    "source": "github_raw",
                    "url": url,
                }
                self._add_record(prompt, code_text, metadata)
                time.sleep(self.request_interval)
            except Exception as exc:  # noqa: BLE001
                print(f"[GitHub] Failed to collect from {url}: {exc}")

    @staticmethod
    def _derive_prompt_from_code(code_text: str, url: str) -> str:
        head = code_text.lstrip()[:4000]
        docstring_match = re.match(r"^[rRuU]{0,2}[\'\"]{3}(.*?)[\'\"]{3}", head, re.DOTALL)
        if docstring_match:
            doc = docstring_match.group(1)
            cleaned = re.sub(r"\s+", " ", doc).strip()
            if cleaned:
                return cleaned

        comment_match = re.match(r"(?:#.*\n)+", head)
        if comment_match:
            comment_text = comment_match.group(0)
            cleaned = re.sub(r"#\s?", "", comment_text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                return cleaned

        filename = url.split("/")[-1]
        fallback = filename.replace("_", " ").replace("-", " ")
        return f"Read and understand the following quantum programming example: {fallback}."

    # ------------------------------------------------------------------
    # StackExchange Q&A
    # ------------------------------------------------------------------
    def collect_stackexchange(
        self,
        *,
        tags: Sequence[str],
        max_questions: int,
        answers_per_question: int,
    ) -> None:
        """Collect data from StackExchange."""
        params = {
            "order": "desc",
            "sort": "votes",
            "tagged": ";".join(tags),
            "site": "quantumcomputing",
            "filter": "withbody",
            "pagesize": max_questions,
        }
        try:
            print("[StackExchange] Fetching question list")
            data = self._fetch(f"{STACKEXCHANGE_API}/questions", params=params, is_json=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[StackExchange] Failed to query questions: {exc}")
            return

        items = data.get("items", []) if isinstance(data, dict) else []
        for question in items:
            question_id = question.get("question_id")
            if not question_id:
                continue
            prompt = f"{question.get('title', '').strip()}\n\n"
            prompt += self._clean_html_text(question.get("body", ""))
            answers = self._fetch_question_answers(question_id, answers_per_question)
            for idx, answer_html in enumerate(answers):
                for block_index, code in enumerate(self._extract_code_blocks(answer_html)):
                    metadata = {
                        "source": "stackexchange",
                        "question_id": str(question_id),
                        "answer_index": str(idx),
                        "code_block_index": str(block_index),
                    }
                    self._add_record(prompt, code, metadata)
            time.sleep(self.request_interval)

    def _fetch_question_answers(self, question_id: int, answers_per_question: int) -> List[str]:
        params = {
            "order": "desc",
            "sort": "votes",
            "site": "quantumcomputing",
            "filter": "withbody",
            "pagesize": answers_per_question,
        }
        try:
            url = f"{STACKEXCHANGE_API}/questions/{question_id}/answers"
            data = self._fetch(url, params=params, is_json=True)
            items = data.get("items", []) if isinstance(data, dict) else []
            return [item.get("body", "") for item in items]
        except Exception as exc:  # noqa: BLE001
            print(f"[StackExchange] Failed to fetch answers for {question_id}: {exc}")
            return []

    @staticmethod
    def _extract_code_blocks(answer_html: str) -> List[str]:
        soup = BeautifulSoup(answer_html, "html.parser")
        blocks: List[str] = []
        for pre in soup.find_all("pre"):
            code_tag = pre.find("code")
            text = code_tag.get_text() if code_tag else pre.get_text()
            cleaned = textwrap.dedent(text).strip()
            if cleaned:
                blocks.append(cleaned)
        return blocks

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def save_jsonl(self, path: str) -> None:
        """Save the collected records to a JSONL file."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            for record in self.records:
                payload = {
                    "prompt": record.prompt,
                    "code": record.code,
                    "metadata": record.metadata,
                }
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"Saved {len(self.records)} records to {path}")


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Collect quantum programming data from multiple sources")
    parser.add_argument(
        "--output",
        type=str,
        default="data/quantum_corpus.jsonl",
        help="The path to the output JSONL file.",
    )
    parser.add_argument(
        "--source",
        choices=("qiskit", "github", "stackexchange"),
        nargs="+",
        default=("qiskit", "github", "stackexchange"),
        help="The types of data sources to enable.",
    )
    parser.add_argument(
        "--min_code_lines",
        type=int,
        default=6,
        help="The minimum number of effective lines of code for a snippet to be retained.",
    )
    parser.add_argument(
        "--stackexchange_questions",
        type=int,
        default=8,
        help="The maximum number of questions to fetch from StackExchange.",
    )
    parser.add_ourgent(
        "--stackexchange_answers",
        type=int,
        default=2,
        help="The number of answers to retain for each question.",
    )
    parser.add_argument(
        "--github_file",
        action="append",
        default=[],
        metavar="URL[|PROMPT]",
        help="Additional GitHub raw files, with an optional custom prompt. Pass multiple times to add multiple files.",
    )
    parser.add_argument(
        "--qiskit_url",
        action="append",
        default=[],
        help="Additional Qiskit tutorial links.",
    )
    return parser.parse_args()


def build_custom_github_resources(cli_values: Sequence[str]) -> List[Dict[str, Optional[str]]]:
    """Build a list of custom GitHub resources from command-line values."""
    resources: List[Dict[str, Optional[str]]] = []
    for item in cli_values:
        if "|" in item:
            url, prompt = item.split("|", 1)
            resources.append({"url": url.strip(), "prompt": prompt.strip()})
        else:
            resources.append({"url": item.strip(), "prompt": None})
    return resources


def main() -> None:
    """The main function for the crawler script."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("Please install the required dependencies: `pip install requests beautifulsoup4`")
        return

    args = parse_cli_args()

    collector = QuantumDataCollector(min_code_lines=args.min_code_lines)

    if "qiskit" in args.source:
        urls = list(DEFAULT_QISKIT_TUTORIALS)
        urls.extend(args.qiskit_url)
        collector.collect_qiskit_textbook(urls)

    if "github" in args.source:
        resources = list(DEFAULT_GITHUB_FILES)
        resources.extend(build_custom_github_resources(args.github_file))
        collector.collect_github_files(resources)

    if "stackexchange" in args.source:
        collector.collect_stackexchange(
            tags=DEFAULT_STACKEXCHANGE_TAGS,
            max_questions=args.stackexchange_questions,
            answers_per_question=args.stackexchange_answers,
        )

    collector.save_jsonl(args.output)


if __name__ == "__main__":
    main()
