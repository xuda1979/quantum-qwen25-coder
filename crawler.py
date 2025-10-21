"""
量子编程数据爬虫示例。

该脚本提供了一个基础框架，用于从公共网站收集量子计算相关的编程样例，例如 Qiskit 教程、量子算法博客等。目的是构建用于微调的高质量问答或代码示例数据集。

由于多数网站可能存在反爬虫机制，本示例仅演示核心流程：
 1. 指定若干包含量子代码的网页 URL；
 2. 下载网页内容并解析，抽取问题描述和示例代码；
 3. 按 JSONL 格式保存，每条记录包含 `prompt` 和 `code` 字段。

请在使用时遵守目标网站的使用条款，适当设置请求间隔，必要时使用代理或添加请求头。对于需要登录授权的数据集，请先获得授权许可。
"""

import json
import os
import time
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup


def fetch_page(url: str) -> str:
    """下载指定 URL 的 HTML 内容。

    Args:
        url: 网页地址。
    Returns:
        HTML 文本。如果请求失败则抛出异常。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_qiskit_tutorial(html: str) -> List[Tuple[str, str]]:
    """解析 Qiskit 教程页面，提取问题描述与代码示例。

    该函数假设每个教程章节包含 `<section>`，其中标题描述问题，随后可能包含 `<pre>` 或 `<code>` 标签的代码。

    Args:
        html: 网页 HTML。
    Returns:
        元组列表，每个元素包含 (prompt, code)。
    """
    soup = BeautifulSoup(html, "html.parser")
    data = []
    # 示例解析规则：根据具体网站结构调整
    sections = soup.find_all("section")
    for sec in sections:
        title = sec.find(["h1", "h2", "h3"])
        code_block = sec.find("pre") or sec.find("code")
        if title and code_block:
            prompt = title.get_text(strip=True)
            code = code_block.get_text()
            data.append((prompt, code))
    return data


def crawl_qiskit_tutorials(urls: List[str], output_file: str) -> None:
    """批量爬取一组 Qiskit 教程链接，并将结果写入 JSONL 文件。

    Args:
        urls: 需要爬取的网页列表。
        output_file: 输出 JSONL 文件路径。
    """
    with open(output_file, "w", encoding="utf-8") as f_out:
        for idx, url in enumerate(urls):
            try:
                print(f"Fetching {url} ({idx+1}/{len(urls)})...")
                html = fetch_page(url)
                records = parse_qiskit_tutorial(html)
                for prompt, code in records:
                    obj = {"prompt": prompt, "code": code}
                    f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                # 礼貌等待，避免请求过于频繁
                time.sleep(1.0)
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")


if __name__ == "__main__":
    # 示例列表，可根据需要扩展
    tutorial_urls = [
        "https://qiskit.org/textbook/ch-ex/foundation/quantum-circuit.html",
        "https://qiskit.org/textbook/ch-algorithms/grover-search.html",
    ]
    os.makedirs("data", exist_ok=True)
    crawl_qiskit_tutorials(tutorial_urls, "data/qiskit_tutorials.jsonl")