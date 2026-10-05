#!/usr/bin/env python3
"""Generate src/beyblade/character_tables.py from Katzer's pages.

Fetches all 32 crystallographic point-group character tables from
gernot-katzers-spice-pages.com, validates the representation-theory sum
rules, and emits a per-class data module keyed by spglib Hermann-Mauguin
symbols.

Run:  python3 scripts/generate_character_tables.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from html import unescape
from pathlib import Path

import requests

BASE = "https://gernot-katzers-spice-pages.com/character_tables/{sch}.html"

# spglib Hermann-Mauguin symbol -> Schoenflies page name
SPGLIB_TO_SCHOENFLIES: dict[str, str] = {
    "1": "C1", "m": "C1h", "-1": "S2",
    "2": "C2", "mm2": "C2v", "2/m": "C2h",
    "4": "C4", "4mm": "C4v", "4/m": "C4h", "-4": "S4",
    "3": "C3", "3m": "C3v", "-6": "C3h", "-3": "S6",
    "222": "D2", "-42m": "D2d", "mmm": "D2h",
    "422": "D4", "4/mmm": "D4h",
    "32": "D3", "-3m": "D3d", "-6m2": "D3h",
    "6": "C6", "6mm": "C6v", "6/m": "C6h",
    "622": "D6", "6/mmm": "D6h",
    "23": "T", "m-3": "Th", "432": "O", "-43m": "Td", "m-3m": "Oh",
}


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.replace("\u2014", "-").replace("\u2212", "-").replace("\xa0", " ")


def parse_page(sch: str) -> dict:
    resp = requests.get(BASE.format(sch=sch), timeout=30)
    resp.raise_for_status()
    html = resp.text
    pre = re.search(r"<PRE[^>]*>(.*?)</PRE>", html, re.S)
    if pre is None:
        raise ValueError(f"{sch}: no <PRE> character table found")
    rows_html = pre.group(1)

    # header: group name + class labels; multiplicities live in class spans
    lines = [ln for ln in rows_html.splitlines() if _clean(unescape(re.sub(r"<[^>]+>", "", ln))).strip()]
    first_row = lines[0]
    header = _clean(unescape(re.sub(r"<[^>]+>", "", first_row))).strip()
    tokens = header.split()[1:]  # drop group name
    classes: list[tuple[str, int]] = []
    pending_mult = 1
    for tok in tokens:
        if tok.isdigit():
            pending_mult = int(tok)
        elif tok in ("<R>", "<p>") or tok.startswith("<"):
            break
        else:
            classes.append((tok, pending_mult))
            pending_mult = 1
    if classes[0][0] != "E":
        raise ValueError(f"{sch}: first class is not E: {classes}")

    irreps: dict[str, list[float]] = {}
    for line in lines[1:]:
        line = _clean(unescape(re.sub(r"<[^>]+>", "", line))).strip()
        if not line:
            continue
        name, rest = line.split(None, 1)
        # trailing explanation prose ("Symmetry of Rotations and ...") ends
        # the table region
        if name == "Symmetry":
            break
        if not re.fullmatch(r"[A-Z][A-Za-z0-9'\"]*", name):
            continue  # e.g. 'Click ...' footnotes
        # footnote marker(s) may sit between the name and the chars
        rest = rest.lstrip("*").strip()
        n_chars = len(classes)
        parts = rest.split(None, n_chars)
        try:
            chars = [float(p) for p in parts[:n_chars]]
        except ValueError:
            raise ValueError(f"{sch}/{name}: cannot parse chars from {parts!r}")
        if name.endswith("*"):
            name = name[:-1].strip()
        # Katzer writes the even irrep as a double-prime ("); the rest of the
        # codebase (symmetry.CHARACTER_TABLES) spells it as two apostrophes.
        name = name.replace('\"', "''")
        if name in irreps:
            raise ValueError(f"{sch}: duplicate irrep {name}")
        irreps[name] = chars

    order = sum(m for _, m in classes)
    _validate(sch, classes, irreps, order)
    return {"classes": classes, "irreps": irreps, "order": order,
            "source": BASE.format(sch=sch)}


def _validate(sch: str, classes, irreps, order: int) -> None:
    for name, chars in irreps.items():
        if len(chars) != len(classes):
            raise ValueError(f"{sch}/{name}: {len(chars)} chars vs {len(classes)} classes")
        if chars[0] <= 0:
            raise ValueError(f"{sch}/{name}: chi(E) = {chars[0]}")
        s = sum(m * c * c for (_, m), c in zip(classes, chars))
        if s not in (order, 2 * order):
            raise ValueError(f"{sch}/{name}: sum rule {s} not h or 2h (h={order})")
    names = list(irreps)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            g = sum(m * x * y for (_, m), x, y in zip(classes, irreps[a], irreps[b]))
            if g != 0:
                raise ValueError(f"{sch}: rows {a},{b} not orthogonal ({g})")
    # rows with row-norm 2h bundle a complex-conjugate pair of 1D irreps:
    # they contribute d^2/2 (two complex 1D irreps) to the dimension sum
    total = 0.0
    for chars in irreps.values():
        norm = sum(m * c * c for (_, m), c in zip(classes, chars))
        total += chars[0] ** 2 / (2 if norm == 2 * order else 1)
    if abs(total - order) > 1e-9:
        raise ValueError(f"{sch}: dimension sum {total} != h {order}")


def main() -> int:
    tables: dict[str, dict] = {}
    for sym, sch in SPGLIB_TO_SCHOENFLIES.items():
        tables[sym] = parse_page(sch)
        print(f"{sym:>6s} ({sch:>4s}): h={tables[sym]['order']:2d} "
              f"{len(tables[sym]['classes'])} classes, "
              f"{len(tables[sym]['irreps'])} irreps")

    out = Path(__file__).parent.parent / "src" / "beyblade" / "character_tables.py"
    lines = [
        '"""Point-group character tables for all 32 crystallographic point groups.',
        "",
        "Generated by scripts/generate_character_tables.py from",
        "gernot-katzers-spice-pages.com (G. Katzer). Keys are spglib",
        "Hermann-Mauguin point-group symbols. Classes follow the source page's",
        "order (E first). Rows whose row-norm equals 2h bundle a",
        "complex-conjugate pair of 1D irreps into one real row; the trace",
        "characters are still exactly what a real phonon spectrum shows.",
        "",
        'Do not edit by hand - regenerate instead."""',
        "",
        "SPGLIB_TO_SCHOENFLIES: dict[str, str] = " + repr(SPGLIB_TO_SCHOENFLIES),
        "",
        "POINT_GROUP_CHARACTER_TABLES: dict[str, dict] = {",
    ]
    for sym in SPGLIB_TO_SCHOENFLIES:
        t = tables[sym]
        lines.append(f"    {sym!r}: {{")
        lines.append(f"        'order': {t['order']},")
        lines.append(f"        'source': {t['source']!r},")
        lines.append(f"        'classes': {t['classes']!r},")
        lines.append("        'irreps': {")
        for name, chars in t["irreps"].items():
            lines.append(f"            {name!r}: {chars!r},")
        lines.append("        },")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
