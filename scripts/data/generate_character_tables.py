#!/usr/bin/env python3
"""Generate src/beyblade/_character_tables_data.py from the httk-symgen dataset.

Source of data: the httk Spacegroup Data Explorer,
https://symdata.anyterial.se/ (CC-BY 4.0; see httk-sym/ATTRIBUTION.md).
Run scripts/data/download_httk_sym.py first, then this script. The
generated module is committed so the package does not need the JSON at
runtime (import must stay cheap, also on Dardel).

Known deviation from Mulliken convention in the source data (corrected
here): httk labels the D3h (hm '-62m') irrep with basis function z as
A1'' and Rz as A2', while Mulliken (Cotton, Herzberg; and the series
C3v A1=z, D3d A2u=z, D6h A2u=z) puts z in A2''. The surrounding groups
follow Mulliken, so the D3h A1''/A2'' labels are swapped in the source.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "httk-sym" / "symmetry_basics.json"
OUT = ROOT / "src" / "beyblade" / "_character_tables_data.py"
EXPLORER = "https://symdata.anyterial.se/pointgroup/"

# Mulliken label corrections for groups where the source data deviates.
# Evidence: basis_linear (z) must sit in A2'' for D3h.
LABEL_FIXES: dict[str, dict[str, str]] = {
    "-62m": {"A1''": "A2''", "A2''": "A1''"},
}

# httk vs spglib Hermann-Mauguin spellings; keys below use spglib's
# spelling (what PointGroup.symbol produces).
SYMBOL_MAP: dict[str, str] = {
    "-62m": "-6m2",
}


def unique_class_labels(classes: list[dict]) -> list[str]:
    """httk reuses e.g. 'sigma_v' for two distinct C2v classes; disambiguate."""
    seen: dict[str, int] = {}
    out = []
    for c in classes:
        name = c["label"]["ascii"]
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}{seen[name]}")
    return out


# Class order for the groups whose characters the pipeline computes per
# operation (defect_frame_operations / _canonical_operations): the class
# order must match the canonical op blocks [E, C3.., C2'.., sigma_h,
# S3.., sigma_v..]. Key = (op_type, 0 if the axis is z else 1). Ties
# (mm2's two mirror classes) are broken by the x-basis (B1) character,
# descending, so the mirror that does NOT flip x comes first -- that is
# the pipeline's mx (mirror normal y). Groups not in this dict keep the
# source class order, which class-level consumers do not depend on.
_CLASS_RANK: dict[str, list[tuple[int, int]]] = {
    "3m": [(1, 0), (3, 0), (-2, 1)],
    "mm2": [(1, 0), (2, 0), (-2, 1)],
    "-6m2": [(1, 0), (3, 0), (2, 1), (-2, 0), (-6, 0), (-2, 1)],
}


def canonical_class_order(hm, pg, sizes, labels, irreps):
    """Reorder classes to the pipeline's canonical op order (in place)."""
    rank = _CLASS_RANK.get(hm)
    if rank is None:
        return sizes, labels
    classes = pg["conjugacy_classes"]
    x_basis = None
    for basis in (["x"], ["x", "y"]):
        x_basis = next(
            (ir["label"] for ir in pg["character_table_real"] if ir.get("basis_linear") == basis),
            None,
        )
        if x_basis is not None:
            break

    def key(i: int) -> tuple:
        c = classes[i]
        k = (c["op_type"], 0 if c["op_axis"] in ([0, 0, 0], [0, 0, 1]) else 1)
        assert k in rank, f"{hm}: unexpected class key {k} for class {labels[i]}"
        pos = rank.index(k)
        tie = -irreps[x_basis][i] if x_basis else 0.0
        return (pos, tie)

    order = sorted(range(len(sizes)), key=key)
    sizes = [sizes[i] for i in order]
    labels = [labels[i] for i in order]
    for name, chi in irreps.items():
        irreps[name] = [chi[i] for i in order]
    return sizes, labels


def check_group(hm: str, order: int, sizes: list[int],
                irreps: dict[str, list[float]]) -> dict[str, bool]:
    """Character orthogonality and dimension sum (class-level).

    Rows that combine complex-conjugate irreps (marked ``complex_pair``)
    have inner product 2*order with themselves instead of ``order``; the
    real decomposition then counts such a pair as one real irrep.
    """
    import numpy as np

    chi = np.asarray(list(irreps.values()), dtype=float)
    s = np.asarray(sizes, dtype=float)
    gram = chi @ (chi * s).T
    diag = np.diag(gram)
    off = gram - np.diagflat(diag)
    assert np.allclose(off, 0.0, atol=1e-10), f"{hm}: rows not orthogonal\n{off}"
    complex_pair = {}
    for name, d in zip(irreps, diag):
        assert np.isclose(d, order) or np.isclose(d, 2 * order), \
            f"{hm}: irrep {name} has wrong inner product {d}"
        complex_pair[name] = bool(np.isclose(d, 2 * order))
    dims = chi[:, 0]
    assert np.allclose(dims, np.round(dims)), f"{hm}: non-integer dimensions"
    total = sum(int(d) ** 2 // (2 if complex_pair[n] else 1) for n, d in zip(irreps, dims))
    assert total == order, f"{hm}: sum of d^2 (with pair counting) != order"
    return complex_pair


def main() -> None:
    version = subprocess.run(
        ["git", "describe", "--always", "--tags"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    data = json.loads(SOURCE.read_text())
    meta = {
        "source": EXPLORER,
        "dataset_version": data.get("version"),
        "dataset_created": data.get("dcterms:created"),
        "license": data.get("dcterms:license"),
        "generated": datetime.now().isoformat(timespec="seconds"),
        "beyblade_version": version,
    }

    groups: dict[str, dict] = {}
    for pg in data["data"]["pointgroups"]:
        hm = SYMBOL_MAP.get(pg["hm_symbol"], pg["hm_symbol"])
        order = pg["order"]
        sizes = [c["size"] for c in pg["conjugacy_classes"]]
        labels = unique_class_labels(pg["conjugacy_classes"])
        irreps: dict[str, dict] = {}
        for ir in pg["character_table_real"]:
            src_hm = pg["hm_symbol"]
            label = LABEL_FIXES.get(src_hm, {}).get(ir["label"], ir["label"])
            irreps[label] = {
                "characters": [float(c) for c in ir["characters"]],
                "dimension": int(ir["dimension"]),
                "basis": {
                    kind: ir[kind]
                    for kind in ("basis_linear", "basis_rotation", "basis_quadratic")
                    if kind in ir
                },
                "label_unicode": ir.get("label_markup", {}).get("unicode", label),
                "label_latex": ir.get("label_markup", {}).get("latex"),
            }
        chars = {name: e["characters"] for name, e in irreps.items()}
        complex_pair = check_group(hm, order, sizes, chars)
        # canonical_class_order reassigns the char lists; sync them back
        # into the irrep dicts.
        sizes, labels = canonical_class_order(hm, pg, sizes, labels, chars)
        for name, chi in chars.items():
            irreps[name]["characters"] = chi
        groups[hm] = {
            "schoenflies": pg["schoenflies"],
            "order": order,
            "crystal_system": pg["crystal_system"],
            "laue_class": pg["laue_class"],
            "is_centrosymmetric": pg["is_centrosymmetric"],
            "classes": labels,
            "class_sizes": sizes,
            "irreps": irreps,
            "complex_pair": complex_pair,
        }
        print(f"{hm:8s} {pg['schoenflies']:5s} order {order:3d} "
              f"classes {len(sizes):2d} irreps {len(irreps):2d}")

    lines = [
        '"""Character tables for the 32 crystallographic point groups.',
        "",
        "GENERATED FILE - do not edit by hand.",
        f"Source: {EXPLORER} (httk-symgen, {meta['dataset_version']},",
        f"created {meta['dataset_created']}), license {meta['license']}.",
        "See httk-sym/ATTRIBUTION.md and scripts/data/generate_character_tables.py.",
        f"Generated {meta['generated']} at beyblade {meta['beyblade_version']}.",
        "",
        "One entry per Hermann-Mauguin point-group symbol (spglib spelling),",
        "class-level: characters are listed once per conjugacy class, in the",
        "order of ``classes`` with multiplicities in ``class_sizes``.",
        '"""',
        "",
        f"# Table of contents: {', '.join(groups)}",
        "",
        "CLASS_TABLES = {",
    ]
    for hm, g in groups.items():
        lines.append(f'    "{hm}": {{')
        lines.append(f'        "schoenflies": {g["schoenflies"]!r},')
        lines.append(f'        "order": {g["order"]},')
        lines.append(f'        "crystal_system": {g["crystal_system"]!r},')
        lines.append(f'        "laue_class": {g["laue_class"]!r},')
        lines.append(f'        "is_centrosymmetric": {g["is_centrosymmetric"]!r},')
        lines.append(f'        "classes": {g["classes"]!r},')
        lines.append(f'        "class_sizes": {g["class_sizes"]!r},')
        lines.append('        "irreps": {')
        for name, e in g["irreps"].items():
            lines.append(f'            "{name}": {{')
            lines.append(f'                "characters": {e["characters"]!r},')
            lines.append(f'                "dimension": {e["dimension"]},')
            for kind, vals in e["basis"].items():
                lines.append(f'                "{kind}": {vals!r},')
            lines.append(f'                "label_unicode": {e["label_unicode"]!r},')
            if e["label_latex"] is not None:
                lines.append(f'                "label_latex": {e["label_latex"]!r},')
            lines.append('            },')
        lines.append("        },")
        lines.append(f'        "complex_pair": {g["complex_pair"]!r},')
        lines.append("    },")
    lines.append("}")
    lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"\nwrote {OUT} ({OUT.stat().st_size / 1e3:.1f} kB)")


if __name__ == "__main__":
    main()
