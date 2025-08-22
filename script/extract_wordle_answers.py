"""
Scrape past Wordle answers from wordlehints.co.uk and write a clean list.

What it does:
- Downloads the page with historical answers.
- Parses visible text and extracts rows like: YYYY-MM-DD (Day) <num> <ANSWER>
- Captures the final 5-letter UPPERCASE token as the answer.
- Lowercases, de-duplicates while preserving calendar order, and writes to file.

Usage:
    python -m script.extract_wordle_answers --out packages/datasets/data/answers_5.txt
    # or alphabetically sorted:
    python -m script.extract_wordle_answers --sort --out packages/datasets/data/answers_5.txt
"""

import re
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup  # pip install beautifulsoup4 requests

URL = "https://wordlehints.co.uk/wordle-past-answers/"
ROW_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*\([A-Za-z]+\)\s*\d+\s+([A-Z]{5})\b")


def unique_preserve_order(words):
    seen = set()
    out = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def fetch_answers(url: str = URL) -> list[str]:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text("\n", strip=True)
    answers = [m.group(2).lower() for m in ROW_RE.finditer(text)]
    return unique_preserve_order(answers)  # <-- removes duplicates, keeps calendar order


def main():
    ap = argparse.ArgumentParser(description="Extract unique Wordle answers")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default="packages/datasets/data/answers_5.txt")
    ap.add_argument("--sort", action="store_true", help="sort alphabetically instead of keeping "
                                                        "calendar order")
    args = ap.parse_args()

    answers = fetch_answers(args.url)
    if args.sort:
        answers = sorted(set(answers))     # optional: alphabetical unique

    Path(args.out).write_text("\n".join(answers) + "\n", encoding="utf-8")
    print(f"Wrote {len(answers)} unique answers -> {args.out}")

if __name__ == "__main__":
    main()
