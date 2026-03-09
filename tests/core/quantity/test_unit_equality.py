from dataclasses import FrozenInstanceError
import pytest

from quantium.core.dimensions import LENGTH, TIME
from quantium.core.unit import LinearUnit
from quantium.units.registry import DEFAULT_REGISTRY as ureg



# -------------------------------
# LinearUnit equality (regressions)
# -------------------------------

@pytest.mark.regression(reason="Issue #24: Units with identical dim & scale must compare equal regardless of name")
def test_unit_equality_ignores_name_and_matches_scale_and_dim():
    # Construct Newton from base units and compare with predefined "N"
    kg = ureg.get("kg")
    m  = ureg.get("m")
    s  = ureg.get("s")
    N1 = kg * m / (s ** 2)   # name "kg·m/s^2", scale 1.0, dim (1,1,-2,0,0,0,0)
    N2 = ureg.get("N")       # name "N",         scale 1.0, dim (1,1,-2,0,0,0,0)

    assert N1 == N2
    # Sanity: equality is determined by dimension + scale; names may or may not match
    assert N1.name in "N"
    assert N1.scale_to_si == N2.scale_to_si
    assert N1.dim == N2.dim


@pytest.mark.regression(reason="Units with different scales are not equal even if dimensions match")
def test_unit_inequality_different_scale_same_dim():
    m  = LinearUnit("m", 1.0, LENGTH)
    cm = LinearUnit("cm", 0.01, LENGTH)
    assert m != cm


@pytest.mark.regression(reason="Units with different dimensions are not equal even if scales match")
def test_unit_inequality_different_dim_same_scale():
    # scale 1.0 but different dimensions (LENGTH vs TIME)
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)
    assert m != s


@pytest.mark.regression(reason="__eq__ should return NotImplemented for incompatible types")
def test_unit_equality_with_incompatible_type_returns_notimplemented():
    m = LinearUnit("m", 1.0, LENGTH)
    # Direct dunder call to observe NotImplemented (== would coerce to False)
    assert LinearUnit.__eq__(m, 42) is NotImplemented
    assert (m == 42) is False


# -------------------------------
# Mixed constructions (extra safety)
# -------------------------------

@pytest.mark.regression(reason="Derived unit equality is stable across different build paths")
def test_unit_equality_across_multiple_construction_paths():
    # (kg·m)/s^2 == (kg/s^2)·m == kg·(m/s^2)
    kg = ureg.get("kg")
    m = ureg.get("m")
    s = ureg.get("s")
    u1 = kg * m / (s ** 2)
    u2 = (kg / (s ** 2)) * m
    u3 = kg * (m / (s ** 2))
    assert u1 == u2 == u3