"""
quantium.core.quantity
======================

Defines the `LinearUnit` and `Quantity` classes for representing and manipulating
physical quantities with units and dimensions in a consistent, SI-based system.

This module provides:
- A `LinearUnit` class for defining physical units (e.g., meter, second, kilogram)
  with their corresponding scaling factors to SI base units and dimensional
  representation.
- A `Quantity` class for representing values with both magnitude and units,
  enabling dimensional arithmetic and automatic unit consistency checks.

The system supports:
- Dimensional analysis and arithmetic operations between quantities.
- Conversion between compatible units.
- Creation of derived quantities via multiplication, division, and exponentiation.
"""



from __future__ import annotations

from math import isclose
from typing import Union

from quantium.core.dimensions import DIM_0, Dim, dim_div, dim_mul, dim_pow
from quantium.core.unit import UNIT_SIMPLIFIER, LinearUnit
from quantium.io.unit_simplifier import SymbolComponents
from quantium.units.parser import extract_unit_expr

Number = Union[int, float]


class Quantity:
    """
    Represents a physical quantity with magnitude, dimension, and unit, supporting
    arithmetic operations and unit conversions while maintaining dimensional consistency.

    Attributes
    ----------
    _mag_si : float
        The magnitude of the quantity expressed in SI base units.
    dim : dict or custom dimension object
        The physical dimension of the quantity (e.g., length, time, mass).
    unit : LinearUnit
        The unit in which the quantity is currently represented.
    """
    __slots__ = ["_mag_si", "dim", "unit"]


    def __init__(self, value : float, unit : LinearUnit):
        self._mag_si = unit.to_base_abs(float(value))
        self.dim = unit.dim
        self.unit = unit

    def _symbol_component_map(self, priority: int) -> SymbolComponents:
        return UNIT_SIMPLIFIER.unit_symbol_map(self.unit, priority)
        
    def _check_dim_compatible(self, other: object) -> None:
        """Internal helper to raise TypeError on dimension mismatch."""
        if not isinstance(other, Quantity):
            # Allow comparison with 0 (dimensionless)
            if isinstance(other, (int, float)) and other == 0:
                if self.dim != DIM_0:
                    raise TypeError("Cannot compare a dimensioned quantity to 0")
                return  # It's a 0 dimensionless quantity, OK
            raise TypeError(f"Cannot compare Quantity with type {type(other)}")

        if self.dim != other.dim:
            raise TypeError(
                f"Cannot compare quantities with different dimensions: "
                f"'{self.unit.name}' and '{other.unit.name}'"
            )
        
    def _is_close(self, other_si_mag: float) -> bool:
        """Internal helper for fuzzy equality."""
        return isclose(self._mag_si, other_si_mag, rel_tol=1e-12, abs_tol=0.0)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        # Same physical dimension; SI magnitudes equal within tolerance.
        return (
            self.dim == other.dim
            and isclose(self._mag_si, other._mag_si, rel_tol=1e-12, abs_tol=0.0)
        )
    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Quantity):
            return NotImplemented
        if self.dim != other.dim:
            return True # Not equal if dims don't match
        return not self._is_close(other._mag_si)
    
    def __lt__(self, other: object) -> bool:
        self._check_dim_compatible(other)
        other_si_mag = getattr(other, '_mag_si', 0.0)
        # Strictly less than AND not fuzzy-equal
        return self._mag_si < other_si_mag and not self._is_close(other_si_mag)

    def __le__(self, other: object) -> bool:
        self._check_dim_compatible(other)
        other_si_mag = getattr(other, '_mag_si', 0.0)
        # Less than OR fuzzy-equal
        return self._mag_si < other_si_mag or self._is_close(other_si_mag)

    def __gt__(self, other: object) -> bool:
        self._check_dim_compatible(other)
        other_si_mag = getattr(other, '_mag_si', 0.0)
        # Strictly greater than AND not fuzzy-equal
        return self._mag_si > other_si_mag and not self._is_close(other_si_mag)

    def __ge__(self, other: object) -> bool:
        self._check_dim_compatible(other)
        other_si_mag = getattr(other, '_mag_si', 0.0)
        # Greater than OR fuzzy-equal
        return self._mag_si > other_si_mag or self._is_close(other_si_mag)
    
    # --- Hashing Solution ---

    def as_key(self, precision: int = 12) -> tuple:
        """
        Returns a hashable, discretized key for this quantity.

        This is the recommended way to use Quantities in dictionaries
        or sets, as it forces the user to choose a precision
        level for "fuzzy" hashing.

        The standard `__hash__` is not implemented because `__eq__`
        uses `isclose`, which would violate the Python hash contract.

        Usage:
        >>> my_dict = {}
        >>> q1 = (1.0 + 1e-13) * u.m
        >>> q2 = (1.0 - 1e-13) * u.m
        >>>
        >>> # q1 and q2 are "equal" but not hash-equal
        >>> q1 == q2  # True
        >>>
        >>> # Using as_key forces them to be hash-equal
        >>> my_dict[q1.as_key(precision=9)] = "value"
        >>> print(my_dict[q2.as_key(precision=9)])
        "value"

        Parameters
        ----------
        precision : int, optional
            The number of decimal places to round the *SI magnitude*
            to for hashing, by default 12 (which is typically
            near 64-bit float precision limits).

        Returns
        -------
        tuple
            A hashable tuple of (dimension, rounded_si_magnitude).
        """
        # Round the SI magnitude to the specified precision
        rounded_mag_si = round(self._mag_si, precision)
        
        # We must also handle -0.0 vs 0.0, which round identically
        # but have different hashes.
        if rounded_mag_si == 0.0:
            rounded_mag_si = 0.0
            
        return (self.dim, rounded_mag_si)

    def to(self, new_unit: "LinearUnit|str") -> Quantity:
        if(isinstance(new_unit, str)):
            from quantium.units.registry import DEFAULT_REGISTRY
            new_unit = extract_unit_expr(new_unit, DEFAULT_REGISTRY)
        
        # This proves to mypy that new_unit is a LinearUnit, not a str.
        if not isinstance(new_unit, LinearUnit):
            raise TypeError(
                "Internal error: unit expression did not resolve to a LinearUnit object."
            )

        if new_unit.dim != self.dim:
            raise TypeError("Dimension mismatch in conversion")
        
        # Optimization: Avoid re-allocating if the target unit is
        # *already* our current unit (same name AND dim).
        # We must check name, as 'V/m' == 'W/(A·m)' is True physically,
        # but the user's intent in to() is to get the new name.
        # The dim check has already passed at this point.
        if new_unit.name == self.unit.name:
            return self

        components = UNIT_SIMPLIFIER.unit_symbol_map(new_unit)
        value, unit = UNIT_SIMPLIFIER.si_to_value_unit(
            self._mag_si,
            self.dim,
            components,
            new_unit,
        )
        return Quantity(value, unit)
        
    
    def to_si(self) -> Quantity:
        """
        Return an equivalent Quantity expressed in SI with a preferred symbol when possible.
        Strategy:
        1) If the current unit clearly belongs to a specific SI family (atomic symbol with
            scale 1, or a prefixed form of one), keep that family in SI (e.g., kBq → Bq).
        2) Otherwise, use the dimension's preferred symbol (A, N, W, Pa, Hz, …).
        3) If no preferred symbol exists, compose the base-SI name from the dimension.
        """
        # Local imports avoid circular import at module load time.
        from quantium.core.utils import format_dim, preferred_symbol_for_dim
        from quantium.units.registry import DEFAULT_REGISTRY as _ureg

        cur_name = self.unit.name

        # --- (1) Preserve the "family" if we can (Hz vs Bq, Gy vs Sv, …) ---
        # Grab all atomic SI heads (scale==1, same dim) registered in the system.
        si_heads = [name for name, u in _ureg.all().items()
                    if u.scale_to_si == 1.0 and u.dim == self.dim and not u.is_delta]

        # If our current unit is exactly one of those heads (e.g., "Bq"), or is a prefixed
        # form ending with the head (e.g., "kBq"), keep that head as the SI symbol.
        for head in si_heads:
            if cur_name == head or cur_name.endswith(head):
                si_unit = LinearUnit(head, 1.0, self.dim)
                return Quantity(self._mag_si, si_unit)  # already SI magnitude

        # --- (2) Fall back to the global preferred symbol for this dimension ---
        sym = preferred_symbol_for_dim(self.dim)  # e.g., "A", "N", "W", "Pa", "Hz", …
        if sym:
            si_unit = LinearUnit(sym, 1.0, self.dim)
            return Quantity(self._mag_si, si_unit)

        # --- (3) Compose from base SI if no named symbol exists ---
        si_name = format_dim(self.dim)  # e.g., "kg·m/s²", "1/s", "m"
        si_unit = LinearUnit(si_name, 1.0, self.dim)
        return Quantity(self._mag_si, si_unit)

    @property
    def si(self) -> Quantity:
        return self.to_si()
    
    @property
    def value(self) -> float:
        return self.unit.from_base_abs(self._mag_si)

    # arithmetic
    def __add__(self, other: Quantity) -> Quantity:
        if self.dim != other.dim:
            raise TypeError("Add requires same dimensions")
        # return in left operand's unit
        sum_si = self._mag_si + other._mag_si
        return Quantity(self.unit.from_base_abs(sum_si), self.unit)
    
    def __sub__(self, other: Quantity) -> Quantity:
        if self.dim != other.dim:
            raise TypeError("Sub requires same dimensions")
        diff_si = self._mag_si - other._mag_si
        return Quantity(self.unit.from_base_abs(diff_si), self.unit)
    
    def __mul__(self, other: "Quantity | LinearUnit | Number") -> "Quantity":
        # scalar × quantity
        if isinstance(other, (int, float)):
            new_si = self._mag_si * float(other)
            return Quantity(self.unit.from_base_abs(new_si), self.unit)

        # quantity × unit
        if isinstance(other, LinearUnit):
            result_mag_si = self._mag_si * other.scale_to_si
            result_dim = dim_mul(self.dim, other.dim)

            # Merge the symbol–exponent maps of self (priority 0) and other (priority 1) 
            # into a combined unit representation
            components = UNIT_SIMPLIFIER.combine_symbol_maps(
                self._symbol_component_map(0),
                UNIT_SIMPLIFIER.unit_symbol_map(other, 1),
            )

            
            value, unit = UNIT_SIMPLIFIER.si_to_value_unit(result_mag_si, result_dim, components)
            return Quantity(value, unit)
        
        # quantity × quantity
        result_mag_si = self._mag_si * other._mag_si
        result_dim = dim_mul(self.dim, other.dim)
        components = UNIT_SIMPLIFIER.combine_symbol_maps(
            self._symbol_component_map(0),
            other._symbol_component_map(1),
        )
        value, unit = UNIT_SIMPLIFIER.si_to_value_unit(result_mag_si, result_dim, components)
        return Quantity(value, unit)

    def __rmul__(self, other: float | int) -> "Quantity":
        # allows 3 * (2 m) -> 6 m
        return self.__mul__(other)

    def __truediv__(self, other: "Quantity | LinearUnit | Number") -> "Quantity":
        # quantity / scalar
        if isinstance(other, (int, float)):
            new_si = self._mag_si / float(other)
            return Quantity(self.unit.from_base_abs(new_si), self.unit)
        
        # quantity / unit
        if isinstance(other, LinearUnit):
            result_mag_si = self._mag_si / other.scale_to_si
            result_dim = dim_div(self.dim, other.dim)
            components = UNIT_SIMPLIFIER.combine_symbol_maps(
                self._symbol_component_map(0),
                UNIT_SIMPLIFIER.scale_symbol_map(UNIT_SIMPLIFIER.unit_symbol_map(other, 1), -1),
            )
            value, unit = UNIT_SIMPLIFIER.si_to_value_unit(result_mag_si, result_dim, components)
            return Quantity(value, unit)

        # quantity / quantity
        result_mag_si = self._mag_si / other._mag_si
        result_dim = dim_div(self.dim, other.dim)
        components = UNIT_SIMPLIFIER.combine_symbol_maps(
            self._symbol_component_map(0),
            UNIT_SIMPLIFIER.scale_symbol_map(other._symbol_component_map(1), -1),
        )
        value, unit = UNIT_SIMPLIFIER.si_to_value_unit(result_mag_si, result_dim, components)
        return Quantity(value, unit)

    def __rtruediv__(self, other: float | int) -> "Quantity":
        # scalar / quantity  -> returns Quantity with inverse dimension
        if not isinstance(other, (int, float)):
            return NotImplemented
        result_dim = dim_div(DIM_0, self.dim)
        result_mag_si = float(other) / self._mag_si
        components = UNIT_SIMPLIFIER.scale_symbol_map(self._symbol_component_map(0), -1)
        value, unit = UNIT_SIMPLIFIER.si_to_value_unit(result_mag_si, result_dim, components)
        return Quantity(value, unit)

    def __pow__(self, n: int) -> "Quantity":
        new_unit = self.unit ** n
        return Quantity(new_unit.from_base_abs(self._mag_si ** n), new_unit)
    
    def __repr__(self) -> str:
        # Local imports avoid cyclic imports; modules are cached after the first time.
        from quantium.core.utils import (
            preferred_symbol_for_dim,
            prettify_unit_name_supers,
        )
        from quantium.units.registry import PREFIXES
        from math import log10, floor  # Added import for math functions

        # Numeric magnitude in the *current* unit
        mag = self._mag_si / self.unit.scale_to_si

        # Dimensionless: print bare number
        if self.dim == DIM_0:
            return f"{mag:.15g}"

        # Start from the user’s unit name (keeps cm/ms etc.), with superscripts & cancellation
        # This is CRITICAL: it cancels "kg·mg/kg" to "mg" *before* we check composition.
        pretty = prettify_unit_name_supers(self.unit.name, cancel=True)

        # CRITICAL: Check if the *prettified* name is composed.
        # This check prevents re-formatting of simple units like "cm", "mg", "Pa", "Bq",
        # which fixes regressions.
        is_composed = any(ch in pretty for ch in ("/", "·", "^"))

        if is_composed:
            # Respect the stored composed unit name for representation.
            # We deliberately avoid performing an additional canonicalisation
            # here so that `Quantity.unit.name` matches the printed output.
            return f"{mag:.15g}" if pretty == "1" else f"{mag:.15g} {pretty}"

        # If the pretty name reduces to "1", show just the number
        # This also handles all non-composed units that skipped the `if` block.
        return f"{mag:.15g}" if pretty == "1" else f"{mag:.15g} {pretty}"


    
    def __format__(self, spec: str) -> str:
        """
        Custom string formatting for Quantity objects.

        The format specifier controls whether the quantity is shown in its
        current unit or converted to SI units before printing.

        Supported specifiers
        --------------------
        "" (empty), or "native"
            Display the quantity in its current unit (default).
        "si"
            Display the quantity converted to SI units.

        Examples
        --------
        >>> v = 1000 @ (ureg.get("cm") / ureg.get("s"))
        >>> f"{v}"           # default: show in current unit (cm/s)
        '1000 cm/s'
        >>> f"{v:native}"      # explicit but same as above
        '1000 cm/s'
        >>> f"{v:si}"        # convert and show in SI (m/s)
        '10 m/s'

        Raises
        ------
        ValueError
            If the format specifier is not one of "", "native", or "si".
        """
        spec = (spec or "").strip().lower()
        if spec in ("", "native"):
            return repr(self)          # current native (default)
        if spec == "si":
            return repr(self.to_si())  # force SI
        raise ValueError("Unknown format spec; use '', 'native', or 'si'")
