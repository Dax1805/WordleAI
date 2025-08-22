"""
Remove duplicate lines from a text file.

Features:
- Preserves original order by default (stable dedupe).
- Optional case-insensitive mode (treat 'TEARS' == 'tears').
- Optional stripping of blank/whitespace-only lines.
- Optional sorting AFTER dedupe (alphabetical); otherwise keep input order.
- Overwrite in place by default, or write to a separate --out path.

Usage:
    python -m script.dedupe_txt --in packages/datasets/data/allowed_5.txt \
        --case-insensitive --strip-blanks
"""

import argparse
from pathlib import Path


def read_lines(p: Path) -> list[str]:
    return [ln.rstrip("\r\n") for ln in p.read_text(encoding="utf-8").splitlines()]


def unique_preserve_order(lines: list[str], key=None) -> list[str]:
    seen, out = set(), []
    for s in lines:
        k = key(s) if key else s
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description="Remove duplicate lines from a text file.")
    ap.add_argument("--in", dest="inp", required=True, help="input .txt file")
    ap.add_argument("--out", dest="out", help="output file (default: overwrite input)")
    ap.add_argument("--case-insensitive", action="store_true", help="treat 'TEARS' and 'tears' as the same")
    ap.add_argument("--strip-blanks", action="store_true", help="drop empty/whitespace-only lines")
    ap.add_argument("--sort", action="store_true", help="sort alphabetically after dedupe (otherwise keep original order)")
    args = ap.parse_args()

    inp = Path(args.inp)
    outp = Path(args.out) if args.out else inp
    if not inp.exists():
        raise FileNotFoundError(inp)

    lines = read_lines(inp)
    if args.strip_blanks:
        lines = [s.strip() for s in lines if s.strip()]

    key = (lambda s: s.lower()) if args.case_insensitive else None
    out = unique_preserve_order(lines, key=key)
    if args.sort:
        out = sorted(out, key=(str.lower if args.case_insensitive else None))

    outp.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Input: {inp} ({len(lines)} lines) → Output: {outp} ({len(out)} unique)")

if __name__ == "__main__":
    main()
