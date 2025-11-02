from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose, isfinite
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from quantium.core.dimensions import DIM_0, Dim, dim_div, dim_mul, dim_pow
from quantium.core.utils import rationalize
from quantium.io.unit_simplifier import UnitNameSimplifier

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from quantium.core.quantity import LinearQuantity


@runtime_checkable
class Unit(Protocol):
    name: str
    dim: Dim

    # Is this unit linear (purely multiplicative) w.r.t. SI?
    @property
    def is_linear(self) -> bool: ...

    # Is this a delta (difference) unit? (only meaningful for linear)
    @property
    def is_delta(self) -> bool: ...

    # # Absolute conversions (apply offset if present)
    def to_base_abs(self, x: float) -> float: ...
    def from_base_abs(self, x: float) -> float: ...

    # Delta conversions (no offset)
    def to_base_delta(self, dx: float) -> float: ...
    def from_base_delta(self, dx: float) -> float: ...



@dataclass(frozen=True, slots=True)
class LinearUnit(Unit):
    """Representation of a physical unit with dimensional metadata."""

    name: str
    scale_to_si: float
    dim: Dim
    _is_delta : bool = False

    def __post_init__(self) -> None:
        if len(self.dim) != 7:
            raise ValueError("dim must be a 7-tuple (L,M,T,I,Θ,N,J)")
        if not (self.scale_to_si > 0 and isfinite(self.scale_to_si)):
            raise ValueError("scale_to_si must be a positive, finite number")
        
    def to_base_abs(self, x: float) -> float:
        """Convert an absolute value in this unit to the SI base unit."""
        return x * self.scale_to_si

    def from_base_abs(self, x: float) -> float:
        """Convert an absolute value from the SI base unit to this unit."""
        return x / self.scale_to_si

    def to_base_delta(self, dx: float) -> float:
        """Convert a difference (delta) in this unit to the SI base unit."""
        return dx * self.scale_to_si

    def from_base_delta(self, dx: float) -> float:
        """Convert a difference (delta) from the SI base unit to this unit."""
        return dx / self.scale_to_si


    
    @classmethod
    def delta(cls, name: str, scale_to_si: float, dim: Dim) -> LinearUnit:
        """Factory for delta (difference) units."""
        return cls(name, scale_to_si, dim, _is_delta=True)

    @property
    def is_linear(self) -> bool:
        return True
    
    @property
    def is_delta(self) -> bool:
        return self._is_delta
    

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LinearUnit):
            return NotImplemented
        return (
            self.dim == other.dim
            and isclose(self.scale_to_si, other.scale_to_si, rel_tol=1e-12, abs_tol=0.0)
        )

    def __rmul__(self, value: float) -> "LinearQuantity":
        from quantium.core.quantity import LinearQuantity

        scalar = float(value)
        if scalar == 0.0:
            mag_si = 0.0
            components = UNIT_SIMPLIFIER.unit_symbol_map(self)
            val, unit = UNIT_SIMPLIFIER.si_to_value_unit(mag_si, self.dim, components)
            return LinearQuantity(val, unit)
        return LinearQuantity(scalar, self)

    def __mul__(self, other: "LinearUnit") -> "LinearUnit":
        new_dim = dim_mul(self.dim, other.dim)
        new_scale = self.scale_to_si * other.scale_to_si

        if (
            self.dim == other.dim
            and isclose(self.scale_to_si, other.scale_to_si, rel_tol=1e-12, abs_tol=0.0)
        ):
            base_name = self.name if self.name else other.name
            new_unit_name = UNIT_SIMPLIFIER.normalize_power_name(f"{base_name}^2")
            return LinearUnit(new_unit_name, new_scale, new_dim)

        components = UNIT_SIMPLIFIER.combine_symbol_maps(
            UNIT_SIMPLIFIER.unit_symbol_map(self, 0),
            UNIT_SIMPLIFIER.unit_symbol_map(other, 1),
        )

        if (
            new_dim != DIM_0
            and isclose(new_scale, 1.0, rel_tol=1e-12, abs_tol=0.0)
            and "1" not in components
        ):
            from quantium.core.utils import preferred_symbol_for_dim

            preferred = preferred_symbol_for_dim(new_dim)
            if preferred:
                return LinearUnit(preferred, 1.0, new_dim)

        new_unit_name = UNIT_SIMPLIFIER.format_unit_components(components)
        if new_dim == DIM_0:
            new_unit_name = ""
        return LinearUnit(new_unit_name, new_scale, new_dim)

    def __truediv__(self, other: "LinearUnit") -> "LinearUnit":
        new_dim = dim_div(self.dim, other.dim)
        new_scale = self.scale_to_si / other.scale_to_si

        components = UNIT_SIMPLIFIER.combine_symbol_maps(
            UNIT_SIMPLIFIER.unit_symbol_map(self, 0),
            UNIT_SIMPLIFIER.scale_symbol_map(UNIT_SIMPLIFIER.unit_symbol_map(other, 1), -1),
        )

        if (
            new_dim != DIM_0
            and isclose(new_scale, 1.0, rel_tol=1e-12, abs_tol=0.0)
            and "1" not in components
        ):
            from quantium.core.utils import preferred_symbol_for_dim

            preferred = preferred_symbol_for_dim(new_dim)
            if preferred:
                return LinearUnit(preferred, 1.0, new_dim)

        new_unit_name = UNIT_SIMPLIFIER.format_unit_components(components)
        if new_dim == DIM_0:
            new_unit_name = ""

        return LinearUnit(new_unit_name, new_scale, new_dim)

    def __rtruediv__(self, n: int | float) -> "LinearUnit":
        if n != 1:
            raise TypeError(
                f"Invalid operation: cannot divide {n} by a Unit ({self.name}). "
                "Only 1/unit (reciprocal) is supported."
            )

        new_dim = dim_div(DIM_0, self.dim)
        components = UNIT_SIMPLIFIER.scale_symbol_map(UNIT_SIMPLIFIER.unit_symbol_map(self, 0), -1)
        new_scale = 1 / self.scale_to_si
        new_name = UNIT_SIMPLIFIER.format_unit_components(components)
        if new_dim == DIM_0:
            new_name = ""
        return LinearUnit(new_name, new_scale, new_dim)

    def __pow__(self, n: int | float | Fraction) -> "LinearUnit":
        if isinstance(n, float):
            n = rationalize(n)

        new_dim = dim_pow(self.dim, n)
        if n == 0:
            normalized_name = ""
        else:
            components = UNIT_SIMPLIFIER.scale_symbol_map(UNIT_SIMPLIFIER.unit_symbol_map(self, 0), n)
            normalized_name = UNIT_SIMPLIFIER.format_unit_components(components)
            if not normalized_name:
                normalized_name = ""

        new_scale = self.scale_to_si ** n
        return LinearUnit(normalized_name, new_scale, new_dim)


UNIT_SIMPLIFIER = UnitNameSimplifier(LinearUnit)

__all__ = ["Unit", "UNIT_SIMPLIFIER"]