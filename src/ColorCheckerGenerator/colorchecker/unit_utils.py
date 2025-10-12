"""
unit_utils.py

Utilities to parse human-friendly dimensions like '200px', '250mm', '25cm', '8in'
and convert them to pixels given a DPI.
"""

import re
from typing import Union
from .config import DEFAULT_DPI

def parse_size(size: Union[str, int, float], dpi: int = DEFAULT_DPI) -> int:
    """
    Parse a size specification and return pixels.

    Args:
        size: e.g. "2000px", "250mm", "25cm", "8in", or an integer (pixels).
        dpi: dots per inch to convert physical units to pixels.

    Returns:
        width in pixels (int).
    """
    if isinstance(size, (int, float)):
        return int(size)
    s = str(size).strip().lower()
    m = re.match(r"^([\d.]+)\s*(px|mm|cm|in)?$", s)
    if not m:
        raise ValueError(f"Invalid size string: {size}")
    val, unit = float(m.group(1)), (m.group(2) or "px")
    if unit == "px":
        return int(val)
    if unit == "mm":
        return int(val / 25.4 * dpi)
    if unit == "cm":
        return int((val * 10) / 25.4 * dpi)
    if unit == "in":
        return int(val * dpi)
    raise ValueError(f"Unsupported unit: {unit}")
