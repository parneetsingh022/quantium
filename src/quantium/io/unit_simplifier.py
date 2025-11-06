"""Utilities for canonical unit name simplification.

This module centralises the logic used to canonicalise and simplify unit names
within the quantity system.  The functionality is wrapped inside
``UnitNameSimplifier`` so it can be reused (and more easily tested) without
keeping a tight coupling to ``quantium.core.quantity``.
"""

from __future__ import annotations

from functools import lru_cache
import math
import re
from typing import Dict, List, Tuple, TYPE_CHECKING
from fractions import Fraction

from quantium.core.dimensions import DIM_0, Dim, dim_pow
from quantium.core.utils import _tokenize_name_merge
from quantium.units.prefixes import Prefix

if TYPE_CHECKING:  # pragma: no cover - import only used for typing
    from quantium.core.unit import LinearUnit, AffineUnit, Unit

SymbolComponents = Dict[str, Tuple[Fraction, Tuple[int, int]]]


_POWER_RE = re.compile(r"^(?P<base>.+?)\^(?P<exp>-?\d+)$")
_MAX_CANON_POWER = 12
_ALLOWED_CANON_PREFIX_SYMBOLS = frozenset({"G", "M","k", "m", "\u00b5", "n", "p"})


def _linearize_for_composition(u: "Unit") -> "LinearUnit":
    """Return a guaranteed LinearUnit for composition algebra.

    Affine units with non-zero offset (°C/°F) are converted to their *delta* unit.
    Ratio-scale affine units (K/°R) retain their symbol but are re-instantiated
    as a LinearUnit so downstream code never has to handle AffineUnit logic.
    """
    from quantium.core.unit import AffineUnit, LinearUnit
    if isinstance(u, AffineUnit):
        du = u.delta_unit  # underlying LinearUnit
        if u.offset_to_si != 0.0:
            return du  # absolute °C/°F → use delta (Δ°C/Δ°F)
        # ratio-scale affine (K / °R): preserve glyph, force linear
        return LinearUnit(name=(u.name or du.name), scale_to_si=du.scale_to_si, dim=du.dim)
    if isinstance(u, LinearUnit):
        return u
    # Defensive: unexpected Unit implementation
    raise TypeError("_linearize_for_composition expected LinearUnit or AffineUnit")


def _dim_key(dim: Dim) -> tuple[int, ...]:
    """Return a hashable key for cached dimension lookups."""
    return tuple(dim)


class UnitNameSimplifier:
    """Encapsulates the heuristics for producing canonical unit names."""

    def __init__(self, unit_cls: type["LinearUnit"]) -> None:
        self._unit_cls = unit_cls

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_power_name(name: str) -> str:
        """Simplifies power expressions like 'x^1' → 'x' and 'x^0' → '1', leaving others unchanged."""
        match = _POWER_RE.match(name)
        if not match:
            return name
        base = match.group("base")
        exp = int(match.group("exp"))
        if exp == 1:
            return base
        if exp == 0:
            return "1"
        return f"{base}^{exp}"

    @staticmethod
    @lru_cache(maxsize=None)
    def _preferred_dim_symbol_map() -> dict[tuple[int, ...], str]:
        """Builds and caches a map from dimension signatures to their preferred SI unit symbols."""
        from quantium.core.utils import preferred_symbol_for_dim
        from quantium.units.registry import DEFAULT_REGISTRY

        mapping: dict[tuple[int, ...], str] = {}
        for _ , unit in DEFAULT_REGISTRY.all().items():
            sym = preferred_symbol_for_dim(unit.dim)
            if sym:
                mapping.setdefault(_dim_key(unit.dim), sym)
        return mapping

    @classmethod
    def _match_preferred_power(cls, dim: Dim) -> tuple[str, int] | None:
        """Detect if ``dim`` is a power of a known preferred symbol dimension."""
        base_map = cls._preferred_dim_symbol_map()
        for base_dim_tuple, symbol in base_map.items():
            base_dim = base_dim_tuple
            if dim_pow(base_dim, -1) == dim:
                return symbol, -1
            for power in range(2, _MAX_CANON_POWER + 1):
                if dim_pow(base_dim, power) == dim:
                    return symbol, power
                if dim_pow(base_dim, -power) == dim:
                    return symbol, -power
        return None

    def canonical_unit_for_dim(self, dim: Dim) -> "LinearUnit":
        """Return a canonical unit (scale 1) for the provided dimension."""
        unit_cls = self._unit_cls
        if dim == DIM_0:
            return unit_cls("", 1.0, DIM_0)

        from quantium.core.utils import format_dim, preferred_symbol_for_dim

        sym = preferred_symbol_for_dim(dim)
        if sym:
            return unit_cls(sym, 1.0, dim)

        match = self._match_preferred_power(dim)
        if match:
            base_sym, power = match
            name = self.normalize_power_name(f"{base_sym}^{power}")
            return unit_cls(name, 1.0, dim)

        name = format_dim(dim)
        if name == "1":
            return unit_cls("", 1.0, DIM_0)
        return unit_cls(name, 1.0, dim)

    # ------------------------------------------------------------------
    # Symbol component utilities
    # ------------------------------------------------------------------
    def unit_symbol_map(self, unit: "Unit", priority: int = 0) -> SymbolComponents:
        """
        Create a mapping from unit symbols to their exponents and metadata.

        This function parses a unit's name into its component symbols and associates each
        with its corresponding exponent (as a `Fraction`) and a tuple of metadata
        `(priority, index)`. The mapping helps represent compound units (e.g., `kg·m^2/s^3`)
        in a structured form that can be used for comparison, simplification, or serialization.

        Examples
        --------
        >>> unit_symbol_map(LinearUnit("kg*m^2/s^3"))
        {
            'kg': (Fraction(1, 1), (0, 0)),
            'm': (Fraction(2, 1), (0, 1)),
            's': (Fraction(-3, 1), (0, 2))
        }

        >>> unit_symbol_map(LinearUnit("N"), priority=1)
        {'N': (Fraction(1, 1), (1, 0))}

        Parameters
        ----------
        unit : LinearUnit
            A `LinearUnit` object whose `name` attribute contains the symbolic representation
            of the unit (e.g., `"kg*m^2/s^3"`).
        priority : int, optional
            A priority value that is stored alongside each symbol to indicate
            its relative importance or origin (default is 0).

        Returns
        -------
        SymbolComponents
            A dictionary mapping each symbol string to a tuple:
            `(Fraction(exponent), (priority, index))`, where:
            - `exponent` is the unit’s power as a `Fraction`
            - `priority` is the given priority value
            - `index` is the order of appearance in the unit string

        Notes
        -----
        - If the unit has no `name`, an empty dictionary is returned.
        - If `_tokenize_name_merge` finds no valid symbols, the entire unit name is treated
        as a single symbol with exponent 1.

        """
        if not unit.name:
            return {}

        raw = list(_tokenize_name_merge(unit.name).items())
        if raw:
            mapping: SymbolComponents = {}
            for idx, (symbol, exponent) in enumerate(raw):
                mapping[symbol] = (Fraction(exponent), (priority, idx))
            return mapping

        return {unit.name: (Fraction(1, 1), (priority, 0))}

    @staticmethod
    def scale_symbol_map(components: SymbolComponents, factor: int | Fraction) -> SymbolComponents:
        factor_frac = Fraction(factor)
        if factor_frac == 1:
            return dict(components)
        scaled: SymbolComponents = {}
        for symbol, (exponent, order) in components.items():
            new_exp = exponent * factor_frac
            if new_exp != 0:
                scaled[symbol] = (new_exp, order)
        return scaled

    @staticmethod
    def combine_symbol_maps(*maps: SymbolComponents) -> SymbolComponents:
        """
        Combine multiple symbol-to-exponent mappings into a single consolidated map.

        This function merges several unit symbol maps (as produced by `unit_symbol_map`)
        by summing the exponents of matching symbols, removing any symbols whose total
        exponent becomes zero, and retaining the symbol entry with the lowest
        `(priority, index)` ordering when duplicates occur.

        The resulting mapping represents the algebraic combination of multiple unit
        expressions (e.g., multiplication or division of compound units).

        Examples
        --------
        >>> map1 = {'m': (Fraction(1), (0, 0)), 's': (Fraction(-2), (0, 1))}
        >>> map2 = {'s': (Fraction(2), (1, 0)), 'kg': (Fraction(1), (1, 1))}
        >>> LinearUnit.combine_symbol_maps(map1, map2)
        {'m': (Fraction(1, 1), (0, 0)), 'kg': (Fraction(1, 1), (1, 1))}

        Parameters
        ----------
        *maps : SymbolComponents
            One or more symbol-to-exponent mappings. Each mapping associates a symbol
            (string) with a tuple `(Fraction(exponent), (priority, index))`.

        Returns
        -------
        SymbolComponents
            A merged mapping where:
            - Exponents of matching symbols are summed.
            - Symbols with a total exponent of zero are removed.
            - For symbols present in multiple maps, the smallest `(priority, index)`
            tuple is retained to preserve preferred ordering and precedence.

        Notes
        -----
        - `priority` values determine precedence between symbols originating from
        different sources (lower values take precedence).
        - `index` values preserve the order of symbols as they appear in the
        original unit definitions.
        - This method is typically used to combine units during multiplication,
        division, or simplification.

        """
        combined: SymbolComponents = {}
        for mapping in maps:
            for symbol, (exponent, order) in mapping.items():
                if exponent == 0:
                    continue
                if symbol in combined:
                    existing_exp, existing_order = combined[symbol]
                    new_exp = existing_exp + exponent
                    if new_exp == 0:
                        combined.pop(symbol, None)
                        continue
                    combined[symbol] = (new_exp, min(existing_order, order))
                else:
                    combined[symbol] = (exponent, order)
        return combined

    @staticmethod
    def format_unit_components(components: SymbolComponents) -> str:
        if not components:
            return ""

        def fmt(sym: str, exp: Fraction) -> str:
            abs_exp = abs(exp)
            if abs_exp == 1:
                return sym
            if abs_exp.denominator == 1:
                return f"{sym}^{abs_exp.numerator}"
            return f"{sym}^({abs_exp.numerator}/{abs_exp.denominator})"

        positives: List[str] = []
        negatives: List[str] = []

        for sym in sorted(components):
            exp, _ = components[sym]
            if exp > 0:
                positives.append(fmt(sym, exp))
            elif exp < 0:
                negatives.append(fmt(sym, exp))

        numerator = "·".join(positives) if positives else "1"
        denominator = "·".join(negatives)

        if denominator:
            if "·" in denominator:
                denominator = f"({denominator})"
            return f"{numerator}/{denominator}"
        return numerator

    @staticmethod
    def _dim_single_axis(dim: Dim) -> tuple[int, Fraction] | None:
        axes = [idx for idx, power in enumerate(dim) if power != 0]
        if len(axes) != 1:
            return None
        axis = axes[0]
        return axis, Fraction(dim[axis])

    def _choose_symbol_for_axis(
        self,
        components: SymbolComponents,
        axis_idx: int,
        target_exp: Fraction,
    ) -> "LinearUnit | AffineUnit | None":
        from quantium.units.registry import DEFAULT_REGISTRY

        target_sign = 1 if target_exp > 0 else -1
        best: tuple[str, Fraction, "LinearUnit | AffineUnit"] | None = None
        best_score: Fraction = Fraction(-1, 1)
        fallback: tuple[str, Fraction, "LinearUnit | AffineUnit"] | None = None
        fallback_score: Fraction = Fraction(-1, 1)

        ordered_items = sorted(components.items(), key=lambda item: item[1][1])

        from quantium.core.unit import LinearUnit, AffineUnit
        for symbol, (exponent, _) in ordered_items:
            if exponent == 0:
                continue
            try:
                candidate = DEFAULT_REGISTRY.get(symbol)
            except ValueError:
                continue
            # Only consider known unit types
            if not isinstance(candidate, (LinearUnit, AffineUnit)):
                continue

            axis_info = self._dim_single_axis(candidate.dim)
            if axis_info is None:
                continue

            cand_axis, _ = axis_info
            if cand_axis != axis_idx:
                continue

            score = abs(exponent)
            entry = (symbol, exponent, candidate)

            if (exponent > 0 and target_sign > 0) or (exponent < 0 and target_sign < 0):
                if score > best_score:
                    best = entry
                    best_score = score
            else:
                if score > fallback_score:
                    fallback = entry
                    fallback_score = score

        chosen = best or fallback
        if not chosen:
            return None

        _, _, unit = chosen
        return unit

    def _unit_from_components(self, components: SymbolComponents) -> "LinearUnit" | None:
        """Reconstruct a composite unit from component symbol exponents."""
        if not components:
            return None

        from quantium.units.registry import DEFAULT_REGISTRY

        def _mul_units(units: List["LinearUnit"]) -> "LinearUnit" | None:
            result: "LinearUnit" | None = None
            for u in units:
                result = u if result is None else result * u
            return result

        numer_parts: List["LinearUnit"] = []
        denom_parts: List["LinearUnit"] = []

        for symbol, (exponent, _) in components.items():
            if exponent == 0:
                continue
            try:
                u = DEFAULT_REGISTRY.get(symbol)
            except ValueError:
                return None
            base = _linearize_for_composition(u)

            abs_exp = abs(exponent)
            unit_part = base if abs_exp == 1 else (base ** abs_exp)
            if exponent > 0:
                numer_parts.append(unit_part)
            else:
                denom_parts.append(unit_part)

        numerator = _mul_units(numer_parts)
        denominator = _mul_units(denom_parts)

        unit_cls = self._unit_cls
        if numerator is None and denominator is None:
            return unit_cls("", 1.0, DIM_0)
        if numerator is None:
            numerator = unit_cls("", 1.0, DIM_0)
        if denominator is None:
            return numerator
        return numerator / denominator

    # ------------------------------------------------------------------
    # Public API used by quantity/unit logic
    # ------------------------------------------------------------------
    def si_to_value_unit(
        self,
        mag_si: float,
        dim: Dim,
        components: SymbolComponents | None = None,
        requested_unit: "LinearUnit" | None = None,
    ) -> tuple[float, "LinearUnit"]:
        """Convert an SI magnitude into a value/unit tuple using canonical units."""
        unit_cls = self._unit_cls

        # checks if it's just a simple unit, i.e. no "*", "/", "^", "·"
        def _is_simple(name: str) -> bool:
            return not any(ch in name for ch in ("*", "/", "^", "·"))

        # Assigns a ranking score to a numeric value based on how close it is to the range [1, 1000).
        # Lower tier (0) and smaller |log10(v)| indicate a more “human-friendly” magnitude.
        def _score_value(v: float) -> tuple[int, float]:
            # tier 0: value in [1, 1000), minimize |log10(v)|; else tier 1
            if v == 0.0:
                return (0, 0.0)
            a = abs(v)
            if 1.0 <= a < 1000.0:
                return (0, abs(math.log10(a)))
            return (1, abs(math.log10(a)))

        # Selects the candidate (value, unit) pair whose numeric value is most “human-friendly”
        # — i.e., within [1, 1000) and closest to 1 on a log scale, using _score_value() as the key.
        def _best(candidates: list[tuple[float, "LinearUnit"]]) -> tuple[float, "LinearUnit"]:
            return min(candidates, key=lambda t: _score_value(t[0]))

        def _registry_get(symbol: str) -> "LinearUnit | None":
            from quantium.units.registry import DEFAULT_REGISTRY
            try:
                u = DEFAULT_REGISTRY.get(symbol)
                # Linearize any affine temperature headers into their linear delta
                return _linearize_for_composition(u)
            except ValueError:
                return None

        def _allowed_prefixes() -> list[Prefix]:
            from quantium.units.registry import PREFIXES
            return [p for p in PREFIXES if p.symbol in _ALLOWED_CANON_PREFIX_SYMBOLS]

        def _value_for(unit: "LinearUnit") -> float:
            return mag_si / unit.scale_to_si

        # Builds a list of (value, unit) pairs for a base symbol and its allowed metric prefixes
        # (e.g., "m" -> "mm", "cm", "km"). If `invert=True`, generates reciprocal (1/unit) forms
        # instead — useful for denominator units like 1/s or 1/m.
        # Example: _prefixed_candidates_for_symbol("s", invert=True)
        #          -> [(value_for_1/s, 1/s), (value_for_1/ms, 1/ms), (value_for_1/ks, 1/ks)]
        def _prefixed_candidates_for_symbol(sym: str, invert: bool = False) -> list[tuple[float, "LinearUnit"]]:
            base = _registry_get(sym)
            if base is None or base.scale_to_si <= 0:
                return []
            cands: list[tuple[float, "LinearUnit"]] = [(_value_for(base if not invert else (base ** -1)), base if not invert else (base ** -1))]
            for p in _allowed_prefixes():
                u = _registry_get(f"{p.symbol}{sym}")
                if u is None:
                    continue

                u = u if not invert else (u ** -1)
                cands.append((_value_for(u), u))
            return cands

        # Attempts to collapse components that all share the same physical dimension into a single unit
        # by summing their exponents. It chooses the base with largest |exponent| (ties -> lower priority/index)
        # and raises it to the total exponent; returns (value, unit) if the result matches the target `dim`.
        # Returns None if any symbol is unknown, dimensions differ, or the summed exponent is zero.
        # Example: ordered = [("cm", Fraction(1), (0,0)), ("m", Fraction(1), (0,1))]  # both length
        #         -> total_exp = 2; base = "cm" (earlier in order); canonical = (cm)^2;
        #            if target dim is L^2, returns (value_for_cm2, cm^2); otherwise None.
        def _collapse_same_dim_components(ordered: list[tuple[str, Fraction, tuple[int, int]]]) -> tuple[float, "LinearUnit"] | None:
            # ordered: [(symbol, exponent, order)]
            component_candidates: list[tuple[str, Fraction, "LinearUnit", tuple[int, int]]] = []
            total_exp = Fraction(0, 1)
            reference_dim: Dim | None = None
            for sym, exp, ordt in ordered:
                u = _registry_get(sym)
                if u is None:
                    return None
                if reference_dim is None:
                    reference_dim = u.dim
                elif u.dim != reference_dim:
                    return None
                component_candidates.append((sym, exp, u, ordt))
                total_exp += exp

            if not component_candidates or total_exp == 0:
                return None

            def _pick_key(entry: tuple[str, Fraction, "LinearUnit", tuple[int, int]]) -> tuple[Fraction, int, int]:
                _, exp, _, ordt = entry
                return (abs(exp), -ordt[0], -ordt[1])

            _, _, ref_u, _ = max(component_candidates, key=_pick_key)
            canonical = ref_u ** total_exp
            if canonical.dim == dim:
                return _value_for(canonical), canonical
            return None

        # Tries to simplify a composite unit into its preferred (canonical) symbol or power form.
        # If the dimension matches a known preferred symbol power (e.g., N·m → J), it uses that.
        # Otherwise, it looks for a preferred base symbol and tests prefixed versions (e.g., Pa → kPa)
        # to find a more readable unit. Falls back to the original composite if no simplification works.
        # Example: for comp = "N·m" and dim = energy, returns (value_for_J, J) since 1 J = 1 N·m.
        def _preferred_collapse_from_components(comp: "LinearUnit", comps: SymbolComponents) -> tuple[float, "LinearUnit"]:
            from quantium.core.utils import preferred_symbol_for_dim
            match = self._match_preferred_power(dim)
            if match:
                base_sym, power = match
                if base_sym in comps:  # only if user referenced this symbol
                    base_unit = _registry_get(base_sym)
                    if base_unit is not None:
                        candidate = base_unit ** power
                        return _value_for(candidate), candidate

            pref_sym = preferred_symbol_for_dim(dim)
            if pref_sym and dim != DIM_0:
                cands = _prefixed_candidates_for_symbol(pref_sym, invert=False)
                if cands:
                    return _best(cands)
            return _value_for(comp), comp

        # Handles simple one-dimensional units (like m, s, or kg) by raising the chosen unit
        # to the given exponent. If it's a base unit (e.g., "m" or "s") and no specific unit
        # was requested, it also checks for better prefixed forms (like "km" or "ms")
        # to make the value more readable.
        # Example: chosen="m", target_exp=1 → tries m, mm, km and picks the best (e.g., 5 km instead of 5000 m)
        def _single_axis_path(target_exp: int | Fraction, chosen: "LinearUnit | AffineUnit | None") -> tuple[float, "LinearUnit"] | None:
            if chosen is None:
                return None
            
            chosen = _linearize_for_composition(chosen)
            
            final_unit = chosen ** target_exp

            # Only attempt prefix optimization when not forced by requested_unit and |exp| == 1,
            # and when the chosen symbol equals the dimension's preferred symbol.
            if requested_unit is None and abs(target_exp) == 1:
                from quantium.core.utils import preferred_symbol_for_dim
                pref_sym = preferred_symbol_for_dim(dim)
                if pref_sym and final_unit.name == pref_sym and dim != DIM_0:
                    invert = (target_exp == -1)
                    cands = [(_value_for(final_unit), final_unit)]
                    cands += _prefixed_candidates_for_symbol(pref_sym, invert=invert)
                    return _best(cands)
            return _value_for(final_unit), final_unit

        # ---------- main logic ----------

        # 1) Dimensionless fast-path
        if dim == DIM_0:
            unit = unit_cls("", 1.0, DIM_0)
            return _value_for(unit), unit

        # 2) Honor a simple requested unit (no operators)
        if requested_unit is not None and _is_simple(requested_unit.name):
            return _value_for(requested_unit), requested_unit

        # 3) Component-driven logic
        if components:
            # 3a) Single-axis shortcut
            single_axis = self._dim_single_axis(dim)
            if single_axis is not None:
                axis_idx, target_exp = single_axis
                chosen = self._choose_symbol_for_axis(components, axis_idx, target_exp)
                got = _single_axis_path(target_exp, chosen)
                if got is not None:
                    return got

            # 3b) Same-dimension components collapse
            ordered_components = sorted(
                ((sym, exp, order) for sym, (exp, order) in components.items() if exp),
                key=lambda item: item[2],
            )
            if ordered_components:
                collapsed = _collapse_same_dim_components(ordered_components)
                if collapsed is not None:
                    return collapsed

            # 3c) Composite fallback + preferred symbol/prefix pass
            composite = self._unit_from_components(components)
            if composite is not None and composite.dim == dim:
                result = _preferred_collapse_from_components(composite, components)
                return result or (_value_for(composite), composite)

        # 4) Final fallback: canonical unit for the dimension
        unit = self.canonical_unit_for_dim(dim)
        return _value_for(unit), unit


__all__ = ["SymbolComponents", "UnitNameSimplifier"]
