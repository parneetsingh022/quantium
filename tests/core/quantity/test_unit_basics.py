from dataclasses import FrozenInstanceError
import pytest
from pytest import approx
from quantium.core.dimensions import LENGTH, TEMPERATURE
from quantium.core.unit import LinearUnit


# -------------------------------
# LinearUnit: construction & validation
# -------------------------------

def test_unit_valid():
    m = LinearUnit("m", 1.0, LENGTH)
    assert m.name == "m"
    assert m.scale_to_si == 1.0
    assert m.dim == LENGTH

def test_unit_invalid_dim_length():
    with pytest.raises(ValueError):
        LinearUnit("bad", 1.0, (1, 0, 0))  # not 7-tuple

@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_unit_invalid_scale(scale):
    with pytest.raises(ValueError):
        LinearUnit("x", scale, LENGTH)

def test_unit_is_frozen_and_slotted():
    m = LinearUnit("m", 1.0, LENGTH)

    # frozen => normal assignment raises FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        m.name = "meter"

    # slots => adding a new attribute should fail (AttributeError or TypeError depending on Python)
    with pytest.raises((AttributeError, TypeError)):
        m.some_new_attr = 42


# -------------------------------
# Conversion hooks (abs/delta)
# -------------------------------

def test_to_from_base_abs_roundtrip_length():
    cm = LinearUnit("cm", 0.01, LENGTH)
    # 100 cm -> 1 m (to_base_abs)
    assert cm.to_base_abs(100.0) == approx(1.0)
    # 1 m -> 100 cm (from_base_abs)
    assert cm.from_base_abs(1.0) == approx(100.0)
    # roundtrip
    x = 42.5
    assert cm.from_base_abs(cm.to_base_abs(x)) == approx(x)

def test_to_from_base_delta_roundtrip_length():
    mm = LinearUnit("mm", 0.001, LENGTH)
    # 500 mm delta -> 0.5 m delta
    assert mm.to_base_delta(500.0) == approx(0.5)
    # 0.5 m delta -> 500 mm delta
    assert mm.from_base_delta(0.5) == approx(500.0)
    # roundtrip
    dx = 7.75
    assert mm.from_base_delta(mm.to_base_delta(dx)) == approx(dx)

def test_abs_vs_delta_same_for_linear_units():
    # For pure linear units, abs and delta behave identically.
    km = LinearUnit("km", 1000.0, LENGTH)
    val = 3.2
    dval = 3.2
    assert km.to_base_abs(val) == approx(km.to_base_delta(dval))
    assert km.from_base_abs(val) == approx(km.from_base_delta(dval))


# -------------------------------
# Delta classmethod + flags
# -------------------------------

def test_delta_classmethod_sets_flag_and_keeps_scale_dim_name():
    dF = LinearUnit.delta("Δ°F", 5/9, TEMPERATURE)
    assert isinstance(dF, LinearUnit)
    assert dF.name == "Δ°F"
    assert dF.scale_to_si == approx(5/9)
    assert dF.dim == TEMPERATURE
    assert dF.is_delta is True
    assert dF.is_linear is True  # still a linear unit (no offset)

def test_delta_unit_conversion_hooks_work():
    # Δ°F: 18 Δ°F == 10 K (since 18 * 5/9 = 10)
    dF = LinearUnit.delta("Δ°F", 5/9, TEMPERATURE)
    assert dF.to_base_delta(18.0) == approx(10.0)
    # back: 10 K delta -> 18 Δ°F
    assert dF.from_base_delta(10.0) == approx(18.0)


# -------------------------------
# Properties
# -------------------------------

def test_is_linear_true_for_normal_and_delta_units():
    m = LinearUnit("m", 1.0, LENGTH)
    dC = LinearUnit.delta("Δ°C", 1.0, TEMPERATURE)
    assert m.is_linear is True
    assert dC.is_linear is True

def test_is_delta_true_only_for_delta_units():
    m = LinearUnit("m", 1.0, LENGTH)
    dC = LinearUnit.delta("Δ°C", 1.0, TEMPERATURE)
    assert m.is_delta is False
    assert dC.is_delta is True