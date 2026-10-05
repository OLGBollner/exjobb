"""Symmetry-group dataclasses over the generated character-table snapshot.

``SymmetryGroup`` is the interface the pipeline consumes: a phonon
spectrum's detected point group resolves to one of these objects, and
irrep matching, projectors and product tables operate on it. The data
comes from the httk-symgen snapshot in ``_character_tables_data.py``
(regenerate with scripts/data/generate_character_tables.py); symmetry
operations are NOT in the snapshot -- they live in the downloaded JSON
(httk-sym/) or are hardcoded per defect frame in defect_frame_operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._character_tables_data import CLASS_TABLES

# spglib and httk-symgen already use identical Hermann-Mauguin
# spellings; no translation needed. Keep a map in place in case a
# future data source differs.
_ALIASES: dict[str, str] = {}


@dataclass(frozen=True)
class Irrep:
    """One irreducible representation of a point group.

    Characters are class-level, aligned with the owning group's
    ``classes`` tuple; multiplicity of each class lives in the group's
    ``class_sizes``.
    """

    name: str
    dimension: int
    characters: tuple[float, ...]
    complex_pair: bool
    basis: dict[str, tuple[str, ...]] = field(default_factory=dict)
    label_unicode: str = ""
    label_latex: str | None = None


@dataclass(frozen=True)
class SymmetryGroup:
    """A crystallographic point group with its character table."""

    hm_symbol: str
    schoenflies: str
    order: int
    crystal_system: str
    laue_class: str
    is_centrosymmetric: bool
    classes: tuple[str, ...]
    class_sizes: tuple[int, ...]
    irreps: tuple[Irrep, ...]
    complex_pairs: frozenset[str]

    @property
    def irrep_names(self) -> tuple[str, ...]:
        return tuple(ir.name for ir in self.irreps)

    def irrep(self, name: str) -> Irrep:
        for ir in self.irreps:
            if ir.name == name:
                return ir
        raise KeyError(f"{self.hm_symbol} has no irrep {name!r}")

    def characters(self, name: str) -> tuple[float, ...]:
        return self.irrep(name).characters

    @property
    def dimensions(self) -> tuple[int, ...]:
        return tuple(ir.dimension for ir in self.irreps)


def _build(hm: str) -> SymmetryGroup:
    e = CLASS_TABLES[hm]
    irreps = tuple(
        Irrep(
            name=name,
            dimension=ir["dimension"],
            characters=tuple(ir["characters"]),
            complex_pair=e["complex_pair"][name],
            basis={
                kind: tuple(vals)
                for kind, vals in ir.items()
                if kind.startswith("basis_")
            },
            label_unicode=ir.get("label_unicode", name),
            label_latex=ir.get("label_latex"),
        )
        for name, ir in e["irreps"].items()
    )
    return SymmetryGroup(
        hm_symbol=hm,
        schoenflies=e["schoenflies"],
        order=e["order"],
        crystal_system=e["crystal_system"],
        laue_class=e["laue_class"],
        is_centrosymmetric=e["is_centrosymmetric"],
        classes=tuple(e["classes"]),
        class_sizes=tuple(e["class_sizes"]),
        irreps=irreps,
        complex_pairs=frozenset(
            n for n, c in e["complex_pair"].items() if c
        ),
    )


_CACHE: dict[str, SymmetryGroup] = {}


def get_symmetry_group(symbol: str) -> SymmetryGroup:
    """Resolve a point-group symbol (spglib or HTTK spelling) to a group.

    Raises KeyError for unknown symbols.
    """
    hm = _ALIASES.get(symbol, symbol)
    if hm not in _CACHE:
        if hm not in CLASS_TABLES:
            raise KeyError(
                f"unknown point group {symbol!r} "
                f"(resolved to {hm!r}); available: {sorted(CLASS_TABLES)}"
            )
        _CACHE[hm] = _build(hm)
    return _CACHE[hm]


def available_groups() -> tuple[str, ...]:
    """All supported Hermann-Mauguin symbols, sorted."""
    return tuple(sorted(CLASS_TABLES))
