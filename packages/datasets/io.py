from __future__ import annotations
from pathlib import Path
from typing import Iterable, List


def read_lines(p: Path | str) -> List[str]:
    """
    Read a UTF-8 text file into a list of lines, stripping trailing CR/LF.
    Raises FileNotFoundError if the path doesn't exist.
    """
    p = Path(p)
    if not p.exists():
        raise FileNotFoundError(p)
    return [ln.rstrip("\r\n") for ln in p.read_text(encoding="utf-8").splitlines()]


def write_lines(lines: Iterable[str], p: Path | str) -> str:
    """
    Write lines to a UTF-8 text file, ensuring a trailing newline.
    Returns the string path written.
    """
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)
