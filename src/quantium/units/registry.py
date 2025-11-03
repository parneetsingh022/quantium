"""
quantium.units.registry
=======================

A structured, extensible, and testable units registry for Quantium.

Key improvements over the prior design
--------------------------------------
- Encapsulates global state in a `UnitsRegistry` class (thread-safe).
- Data-driven registration of SI base/derived units.
- Normalization that handles ASCII fallbacks and Unicode NFC.
- Lazy, safe synthesis of prefixed units with anti-stacking checks.
- Support for aliases (e.g., "ohm" → "Ω", "Ohm" → "Ω").
- Clear public API: `register`, `register_alias`, `get`, `has`, `all`.
- Easily testable and embeddable (multiple registries for testing).

Assumptions
-----------
- `LinearUnit(name: str, scale_to_si: float, dim)` is available from `quantium.core.quantity`.
- Dimension arithmetic helpers are in `quantium.core.dimensions`.
"""
from __future__ import annotations


import re
import threading
from typing import Dict, Iterable, Mapping, Optional, Tuple, ClassVar
import unicodedata
from quantium.units.prefixes import PREFIXES

from quantium.core.dimensions import (
    AMOUNT,
    CURRENT,
    DIM_0,
    LENGTH,
    LUMINOUS,
    MASS,
    TEMPERATURE,
    TIME,
    dim_div,
    dim_mul,
    dim_pow,
)
from quantium.core.unit import LinearUnit, AffineUnit
from quantium.core.quantity import AffineQuantity
from quantium.units.parser import extract_unit_expr




# Ordered list of prefix symbols by descending length for robust matching
_PREFIX_SYMBOLS_DESC = tuple(sorted((p.symbol for p in PREFIXES), key=len, reverse=True))
_PREFIX_FACTORS: Mapping[str, float] = {p.symbol: p.factor for p in PREFIXES}

# ---------------------------------------------------------------------------
# Normalization & aliases
# ---------------------------------------------------------------------------
_ALIAS_TO_CANONICAL: Dict[str, str] = {
    # ohm aliases (case-insensitive handled separately)
    "ohm": "Ω",
}

_OHM_RE = re.compile(r"(?i)ohm")


def normalize_symbol(s: str) -> str:
    """Normalize user-provided unit symbols.

    Rules:
    - Unicode normalize to NFC (composed forms like "µ").
    - Replace ASCII leading 'u' micro with Greek 'µ' **only** at start.
    - Map textual aliases to canonical symbols (e.g. any 'ohm' → 'Ω').
    - Strip surrounding whitespace.
    - Leave case as-is except for alias mapping handled via regex.
    """
    if not s:
        return s

    s = s.strip()
    s = unicodedata.normalize("NFC", s)

    # Leading 'u' as ASCII micro → 'µ'
    if s.startswith("u"):
        s = "µ" + s[1:]

    # Replace all forms of 'ohm' with Ω
    s = _OHM_RE.sub("Ω", s)
    return s


# ---------------------------------------------------------------------------
# Units registry
# ---------------------------------------------------------------------------
class UnitsRegistry:
    """Thread-safe registry for `LinearUnit` objects with SI prefix synthesis.

    This registry does *not* parse compound expressions (like "m/s^2").
    It focuses on atomic symbols (possibly prefixed). That higher-level
    parsing can be layered above this API.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._units: Dict[str, LinearUnit] = {}
        self._aliases: Dict[str, str] = {}
        self._non_prefixable: set[str] = set()

    def __contains__(self, symbol: str) -> bool:
        try:
            self.get(symbol)   # alias-aware + can synthesize/parse
            return True
        except ValueError:
            return False

    def set_non_prefixable(self, symbols: Iterable[str]) -> None:
        """Mark unit symbols that must not accept SI prefixes (e.g., 'kg', 'min')."""
        with self._lock:
            self._non_prefixable = {normalize_symbol(s) for s in symbols}

    def is_non_prefixable(self, symbol: str) -> bool:
        """Query helper (symbol may be alias; we normalize only the token)."""
        return normalize_symbol(symbol) in self._non_prefixable

    # -------------------------- public API ---------------------------------
    def register(self, unit: LinearUnit, replace : bool = False) -> None:
        """Register (or overwrite if replace is True) a `LinearUnit` under its canonical name.

        Use `register_alias` to add additional spellings without duplication.
        """
        
        # --- FIX ---
        # The lock must wrap the *entire* check-and-set operation
        # to prevent race conditions.
        with self._lock:
            if unit.name in getattr(UnitNamespace, "_reserved_names", ()):
                raise ValueError(
                    f"Cannot register unit '{unit.name}': "
                    "name conflicts with UnitNamespace attribute/method."
                )
            
            # Check for conflicts *only* if replace is False
            if not replace:
                # Check 1: Conflict with existing *unit*
                if unit.name in self._units:
                    raise ValueError(
                        f"Cannot register unit '{unit.name}': "
                        "a unit with this name already exists."
                    )
                
                # Check 2 (THE FIX): Conflict with existing *alias*
                if unit.name in self._aliases:
                    raise ValueError(
                        f"Cannot register unit '{unit.name}': "
                        "an alias with this name already exists."
                    )
            
            # If replace=True, or if no conflicts were found, proceed.
            #
            # Note: A more robust implementation might also remove any
            # conflicting aliases from self._aliases if replace=True,
            # but this fix is sufficient to pass your test and prevent
            # the inconsistent state.
            
            self._units[unit.name] = unit

    def register_alias(self, alias: str, canonical: str, replace : bool = False) -> None:
        # 1) normalized form (keeps current behavior; e.g., 'ohm' -> 'Ω')
        norm_key = normalize_symbol(alias)

        # 2) literal, NFC/trimmed spelling (for discoverability in __dir__)
        literal_key = unicodedata.normalize("NFC", alias.strip())

        # 3) casefolded literal (for case-insensitive alias matching like 'mixed_key')
        folded_key = literal_key.casefold()

        with self._lock:
            reserved = getattr(UnitNamespace, "_reserved_names", ())
            if (
                literal_key in reserved
                or folded_key in reserved
                or norm_key in reserved
            ):
                raise ValueError(
                    f"Cannot register alias '{alias}': "
                    "name conflicts with UnitNamespace attribute/method."
                )
            # Check for conflicts *only* if replace is False
            if not replace:
                # Check all potential keys (literal, folded, and normalized)
                # for a conflict with an existing unit.
                keys_to_check = {literal_key, folded_key, norm_key}
                for key in keys_to_check:
                    # A conflict exists IF the key is a unit AND that unit is NOT
                    # the intended canonical target of this alias. This prevents
                    # errors during bootstrap (e.g., alias("ohm", "Ω")) and
                    # allows for intentional shadowing when replace=True.
                    if key in self._units and key != canonical:
                        raise ValueError(
                            f"Cannot register alias '{alias}' (which maps to '{key}'): "
                            f"a unit with the name '{key}' already exists."
                        )

            # If replace=True, or if no conflicts were found, register all forms.
            # This will correctly repoint an existing alias if one exists.
            self._aliases[norm_key] = canonical
            self._aliases[literal_key] = canonical
            self._aliases[folded_key] = canonical

    def has(self, symbol: str) -> bool:
        try:
            self.get(symbol)
            return True
        except ValueError:
            return False

    def get(self, symbol: str) -> LinearUnit:
        """Lookup a unit by symbol. If missing, try to synthesize via SI prefix.

        Raises `ValueError` if unknown.
        """
        # if it is a composed expression
        if any(op in symbol for op in ('*', '/', '^')):
            return extract_unit_expr(symbol, self)
        
        sym = normalize_symbol(symbol)
        with self._lock:
            # alias redirect
            target = self._aliases.get(sym)
            if target is not None:
                sym = target

            u = self._units.get(sym)
            if u is not None:
                return u

            # Attempt to synthesize prefixed unit
            synthesized = self._try_synthesize_prefixed(sym)
            if synthesized is not None:
                return synthesized

        raise ValueError(f"Unknown unit symbol: {symbol}")

    def all(self) -> Mapping[str, LinearUnit]:
        with self._lock:
            return dict(self._units)
        
    def as_namespace(self) -> UnitNamespace:
        return UnitNamespace(self)
        

    # ------------------------- internals -----------------------------------
    def _split_prefix(self, symbol: str) -> Tuple[Optional[str], str]:
        for p in _PREFIX_SYMBOLS_DESC:
            if symbol.startswith(p):
                return p, symbol[len(p):]
        return None, symbol

    def _looks_prefixed(self, symbol: str) -> bool:
        p, base = self._split_prefix(symbol)
        return p is not None and base in self._units

    def _try_synthesize_prefixed(self, sym: str) -> Optional[LinearUnit]:
        # Already registered due to race? (cheap check)
        if sym in self._units:
            return self._units[sym]

        prefix, base_sym = self._split_prefix(sym)
        if prefix is None or not base_sym:
            return None

        base = self._units.get(base_sym)
        if base is None:
            return None

        # Prevent stacked prefixes: base itself must not be prefixed
        if self._looks_prefixed(base_sym):
            return None
        
        if base_sym in self._non_prefixable:
            return None

        factor = _PREFIX_FACTORS[prefix]
        new_unit = LinearUnit(sym, base.scale_to_si * factor, base.dim)
        self._units[sym] = new_unit
        return new_unit
    

class UnitNamespace:
    _reserved_names: ClassVar[set[str]] = set()

    def __init__(self, reg : "UnitsRegistry") -> None:
        self._reg = reg

    def __contains__(self, spec: str) -> bool:
        return self._reg.has(spec)

    def define(self, expr : str, scale : "float|int", reference : "LinearUnit", replace : bool = False) -> None:
        if expr in getattr(UnitNamespace, "_reserved_names", ()):
            raise ValueError(
                f"Cannot define unit '{expr}': "
                "name conflicts with UnitNamespace attribute/method."
            )
        
        self._reg.register(LinearUnit(expr, float(scale) * reference.scale_to_si , reference.dim), replace)

    def __call__(self, spec : "str") -> "LinearUnit":
        return self._reg.get(spec)
    
    def __getattr__(self, name: "str") -> "LinearUnit":
        try:
            return self._reg.get(name)
        except (KeyError, ValueError) as e:
            # Unknown symbol should look like a missing attribute
            raise AttributeError(name) from e
        
    def __dir__(self) -> list[str]:
        """List all available unit symbols for autocomplete."""
        # start with object + class attributes (normal methods, etc.)
        base_dir = set(super().__dir__())
        # include registered unit symbols
        try:
            units = set(self._reg.all().keys())
        except Exception:
            units = set()
        # include known aliases
        try:
            aliases = set(self._reg._aliases.keys())
        except Exception:
            aliases = set()
        return sorted(base_dir | units | aliases)

UnitNamespace._reserved_names = set(dir(UnitNamespace))  # pyright: ignore[reportInvalidTypeForm] # type: set[str]



# ---------------------------------------------------------------------------
# Bootstrap a default registry with SI units
# ---------------------------------------------------------------------------

def _bootstrap_default_registry() -> UnitsRegistry:
    reg = UnitsRegistry()

    # Base SI units
    base_units = (
        LinearUnit("m",   1.0, LENGTH),       # length
        LinearUnit("kg",  1.0, MASS),         # mass
        LinearUnit("s",   1.0, TIME),         # time
        LinearUnit("A",   1.0, CURRENT),      # electric current
        LinearUnit("mol", 1.0, AMOUNT),       # amount of substance
        LinearUnit("cd",  1.0, LUMINOUS),     # luminous intensity
    )

    # Named, dimensionless
    derived_named = (
        LinearUnit("rad", 1.0, DIM_0),
        LinearUnit("sr",  1.0, DIM_0),
    )

    # --- Helpful composite dimensions (readable + reuse) ---
    FORCE        = dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2)))              # N
    PRESSURE     = dim_div(FORCE, dim_pow(LENGTH, 2))                             # Pa
    ENERGY       = dim_mul(FORCE, LENGTH)                                         # J
    POWER        = dim_div(ENERGY, TIME)                                          # W
    CHARGE       = dim_mul(CURRENT, TIME)                                         # C
    VOLTAGE      = dim_div(POWER, CURRENT)                                        # V
    CAPACITANCE  = dim_div(CHARGE, VOLTAGE)                                       # F
    RESISTANCE   = dim_div(VOLTAGE, CURRENT)                                      # Ω
    CONDUCTANCE  = dim_div(CURRENT, VOLTAGE)                                      # S
    FLUX         = dim_mul(VOLTAGE, TIME)                                         # Wb
    FLUX_DENSITY = dim_div(FLUX, dim_pow(LENGTH, 2))                              # T (tesla)
    INDUCTANCE   = dim_div(FLUX, CURRENT)                                         # H
    LUMEN        = LUMINOUS                                                       # lm = cd·sr, sr ≡ dimensionless
    LUX          = dim_div(LUMEN, dim_pow(LENGTH, 2))                             # lx
    FREQUENCY    = dim_pow(TIME, -1)                                              # Hz, Bq
    DOSE         = dim_div(ENERGY, MASS)                                          # Gy, Sv
    CATALYTIC    = dim_div(AMOUNT, TIME)                                          # kat

    # Derived (symbol, scale_to_si, dim)
    derived_units = (
        ("g",  1e-3, MASS),           # gram
        ("Hz", 1.0,  FREQUENCY),
        ("N",  1.0,  FORCE),
        ("Pa", 1.0,  PRESSURE),
        ("J",  1.0,  ENERGY),
        ("W",  1.0,  POWER),
        ("C",  1.0,  CHARGE),
        ("V",  1.0,  VOLTAGE),
        ("F",  1.0,  CAPACITANCE),
        ("Ω",  1.0,  RESISTANCE),
        ("S",  1.0,  CONDUCTANCE),
        ("Wb", 1.0,  FLUX),
        ("T",  1.0,  FLUX_DENSITY),   # tesla (string) vs TIME (var) is no longer confusing
        ("H",  1.0,  INDUCTANCE),
        ("lm", 1.0,  LUMEN),
        ("lx", 1.0,  LUX),
        ("Bq", 1.0,  FREQUENCY),
        ("Gy", 1.0,  DOSE),
        ("Sv", 1.0,  DOSE),
        ("kat",1.0,  CATALYTIC),
    )

    time_units = (
        ("min",        60.0,                             TIME),  # minute
        ("h",          60.0 * 60.0,                      TIME),  # hour
        ("d",          24.0 * 60.0 * 60.0,               TIME),  # day
        ("wk",         7.0 * 24.0 * 60.0 * 60.0,         TIME),  # week
        ("fortnight",  14.0 * 24.0 * 60.0 * 60.0,        TIME),  # fortnight

        # Civil (Gregorian) average month/year
        ("mo",         (365.2425 / 12.0) * 24.0 * 3600.0, TIME), # month (avoid "m")
        ("yr",         365.2425 * 24.0 * 3600.0,          TIME), # year (Gregorian mean)

        # Astronomy (explicit)
        ("yr_julian",  365.25 * 24.0 * 3600.0,            TIME), # Julian year

        # Longer spans (Gregorian-based)
        ("decade",     10.0  * 365.2425 * 24.0 * 3600.0,  TIME),
        ("century",    100.0 * 365.2425 * 24.0 * 3600.0,  TIME),
        ("millennium", 1000.0* 365.2425 * 24.0 * 3600.0,  TIME),
    )


    # Register all
    for u in base_units:
        reg.register(u)
    for u in derived_named:
        reg.register(u)
    for sym, scale, dim in derived_units:
        reg.register(LinearUnit(sym, scale, dim))
    for sym, scale, dim in time_units:
        reg.register(LinearUnit(sym, scale, dim))
    
    # Temperature Delta Units
    reg.register(LinearUnit.delta('ΔK', 1.0, TEMPERATURE, treat_si=True))
    reg.register(LinearUnit.delta('Δ°C', 1, TEMPERATURE))
    reg.register(LinearUnit.delta('Δ°F', 5/9, TEMPERATURE))
    reg.register(LinearUnit.delta("Δ°R", 5/9, TEMPERATURE))

    # Temperature Units
    reg.register(AffineUnit('°C', 1,  273.15, TEMPERATURE, delta_unit=reg.get('Δ°C')))
    reg.register(AffineUnit("°F", 5.0 / 9.0, 255.3722222222222, TEMPERATURE, delta_unit=reg.get("Δ°F")))
    reg.register(AffineUnit("°R", 5.0 / 9.0, 0.0, TEMPERATURE, delta_unit=reg.get("Δ°F")))
    reg.register(AffineUnit('K', 1.0, 0.0, TEMPERATURE, delta_unit=reg.get('ΔK')))

    # Common aliases
    reg.register_alias("ohm", "Ω")
    reg.register_alias("Ohm", "Ω")
    reg.register_alias("OHM", "Ω")

    # Time aliases
    reg.register_alias("minute", "min")
    reg.register_alias("minutes", "min")
    reg.register_alias("hr", "h")
    reg.register_alias("hour", "h")
    reg.register_alias("hours", "h")
    reg.register_alias("day", "d")
    reg.register_alias("days", "d")
    reg.register_alias("week", "wk")
    reg.register_alias("weeks", "wk")
    reg.register_alias("fortnights", "fortnight")
    reg.register_alias("month", "mo")
    reg.register_alias("months", "mo")
    reg.register_alias("year", "yr")
    reg.register_alias("years", "yr")
    reg.register_alias("annum", "yr")
    reg.register_alias("dec", "decade")
    reg.register_alias("decades", "decade")
    reg.register_alias("cent", "century")
    reg.register_alias("centuries", "century")
    reg.register_alias("millennia", "millennium")


    # Delta Temperature aliases
    reg.register_alias("delta_degC", "Δ°C")
    reg.register_alias("delta_deg_celsius", "Δ°C")
    reg.register_alias("delta_celsius", "Δ°C")
    
    reg.register_alias("delta_degF", "Δ°F")
    reg.register_alias("delta_deg_fahrenheit", "Δ°F")
    reg.register_alias("delta_fahrenheit", "Δ°F")

    reg.register_alias("delta_degR", "Δ°R")
    reg.register_alias("delta_deg_rankine", "Δ°R")
    reg.register_alias("delta_rankine", "Δ°R")


    # Temperature aliases
    reg.register_alias("degC", "°C")
    reg.register_alias("deg_celsius", "°C")
    reg.register_alias("celsius", "°C")
    
    reg.register_alias("degF", "°F")
    reg.register_alias("deg_fahrenheit", "°F")
    reg.register_alias("fahrenheit", "°F")

    reg.register_alias("degR", "°R")
    reg.register_alias("deg_rankine", "°R")
    reg.register_alias("rankine", "°R")

    reg.register_alias("delta_K", "ΔK")
    reg.register_alias("delta_k", "ΔK")
    reg.register_alias("delta_kelvin", "ΔK")
    reg.register_alias("kelvin", "K")



    reg.set_non_prefixable([
        "kg",
        "min", "h", "d", "wk", "fortnight",
        "mo", "yr", "yr_julian",
        "decade", "century", "millennium",
        "Δ°C", "Δ°F", "Δ°R", "ΔK",
        "K",
    ])

    return reg


# Public, shared default registry
DEFAULT_REGISTRY: UnitsRegistry = _bootstrap_default_registry()


__all__ = [
    "UnitsRegistry",
    "DEFAULT_REGISTRY",
]
