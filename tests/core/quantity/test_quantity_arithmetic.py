import math

import pytest

from quantium.core.dimensions import DIM_0, TIME, LENGTH, dim_div, dim_mul
from quantium.core.quantity import LinearQuantity
from quantium.core.unit import LinearUnit
from quantium.units import u


# -------------------------------
# Arithmetic: +, -, *, /, **, scalars
# -------------------------------

def test_add_and_sub_same_dim():
    m = LinearUnit("m", 1.0, LENGTH)
    cm = LinearUnit("cm", 0.01, LENGTH)
    q1 = 1 * m
    q2 = 50 * cm  # 0.5 m

    s = q1 + q2   # left unit ("m") retained
    d = q1 - q2

    assert s.unit is m and d.unit is m
    assert math.isclose(s._mag_si / s.unit.scale_to_si, 1.5)
    assert math.isclose(d._mag_si / d.unit.scale_to_si, 0.5)

def test_add_dim_mismatch_raises():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)
    with pytest.raises(TypeError):
        _ = (1 * m) + (1 * s)

def test_sub_dim_mismatch_raises():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)
    with pytest.raises(TypeError):
        _ = (1 * m) - (1 * s)

def test_scalar_multiplication_and_division():
    m = LinearUnit("m", 1.0, LENGTH)
    q = 2 * m

    q2 = q * 3
    q3 = 3 * q
    q4 = q / 2

    assert q2.dim == LENGTH and q3.dim == LENGTH and q4.dim == LENGTH
    assert math.isclose(q2._mag_si / q2.unit.scale_to_si, 6.0)
    assert math.isclose(q3._mag_si / q3.unit.scale_to_si, 6.0)
    assert math.isclose(q4._mag_si / q4.unit.scale_to_si, 1.0)

def test_quantity_times_quantity():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)
    q = (2 * m) * (3 * s)  # -> 6 m·s

    assert q.dim == dim_mul(LENGTH, TIME)
    assert q.unit.name == "m·s"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 6.0)

def test_quantity_div_quantity():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)
    q = (10 * m) / (2 * s)  # -> 5 m/s

    assert q.dim == dim_div(LENGTH, TIME)
    assert q.unit.name == "m/s"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 5.0)

def test_scalar_divided_by_quantity():
    m = LinearUnit("m", 1.0, LENGTH)
    q = 2 / (2 * m)  # -> 1 (1/m)

    assert q.dim == dim_div(DIM_0, LENGTH)
    assert q.unit.name == "1/m"
    assert math.isclose(q._mag_si / q.unit.scale_to_si, 1.0)



# -------------------------------
# LinearQuantity * LinearUnit
# -------------------------------

def test_quantity_times_unit_basic():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)

    q = (2 * m) * s  # → 2 m·s

    assert isinstance(q, LinearQuantity)
    assert q.dim == dim_mul(LENGTH, TIME)
    assert q.unit.name == "m·s"
    assert math.isclose(q.value, 2.0)


def test_quantity_times_unit_with_prefix_does_not_change_numeric_value():
    cm = LinearUnit("cm", 0.01, LENGTH)
    m = LinearUnit("m", 1.0, LENGTH)

    q_cm = 300 * cm   # SI = 3.0 m
    q_m  = 200 * m
    out = q_cm * m    # retains prefixed symbol composition
    out2 = q_m * q_cm

    assert out.dim == dim_mul(LENGTH, LENGTH)
    assert out.unit.name == "cm^2"
    assert out2.dim == dim_mul(LENGTH, LENGTH)
    assert out2.unit.name == "m^2"
    assert math.isclose(out.value, 30000)
    assert math.isclose(out2.value, 600)



def test_quantity_times_unit_does_not_mutate_original():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)

    q = 5 * m
    _ = q * s

    # original unchanged
    assert q.unit is m
    assert math.isclose(q._mag_si, 5.0)


# -------------------------------
# LinearQuantity / LinearUnit
# -------------------------------

def test_quantity_div_unit_basic():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)

    q = (10 * m) / s  # → 10 m/s

    assert isinstance(q, LinearQuantity)
    assert q.dim == dim_div(LENGTH, TIME)
    assert q.unit.name == "m/s"
    assert math.isclose(q.value, 10.0)


def test_quantity_div_unit_dimensionless_normalization():
    """Division by a matching unit collapses to a truly dimensionless result."""
    m = LinearUnit("m", 1.0, LENGTH)

    q = (7 * m) / m

    assert q.dim == DIM_0
    assert q.unit.name == ""
    assert math.isclose(q.value, 7.0)



def test_quantity_div_unit_with_prefix_dimensionless_value_is_correct():
    """Prefixed ratios still collapse to a pure, unitless number."""
    cm = LinearUnit("cm", 0.01, LENGTH)
    m = LinearUnit("m", 1.0, LENGTH)

    q = (200 * cm) / m  # 200 cm / 1 m -> 2 (dimensionless)

    assert q.dim == DIM_0
    assert q.unit.name == ""
    assert math.isclose(q.value, 2.0)



def test_quantity_div_unit_does_not_mutate_original():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)

    q = 12 * m
    _ = q / s

    # original unchanged
    assert q.unit is m
    assert math.isclose(q._mag_si, 12.0)


def test_length_only_simplifies_and_allows_custom_conversion():
    q = (4 * (u.m ** 2)) / (4 * (u.cm ** 3))

    assert q.unit.name == "1/cm"
    assert math.isclose(q.value, 10_000.0)

    converted = q.to("cm**2/m**3")
    assert converted.unit.name == "1/m"
    assert math.isclose(converted.value, 1000000.0)


def test_force_substitution_uses_newton_with_prefix_conversion():
    q = (1 * u.kg) * u.mm / (u.s ** 2)

    assert q.unit.name == "mN"
    assert q.value == 1
    assert math.isclose(q.value, 1.0, rel_tol=1e-12)


def test_force_substitution_handles_submicron_inputs():
    q = (1 * u.mg) * u.um / (u.s ** 2)

    assert q.unit.name == "pN"
    assert q.value, 1e-12 == 1
    assert math.isclose(q.value, 1.0, rel_tol=1e-12)


def test_force_substitution_detects_large_prefix():
    q = (1 * u.kg) * u.m / (u.ns ** 2)

    assert q.unit.name == "GN"
    assert math.isclose(q.value, 1_000_000_000)

# --- Tests for Issue #67 ---

@pytest.mark.regression(reason="Issue #67: Operator precedence bug in scalar * unit / unit")
def test_regression_67_scalar_times_unit_div_unit_precedence():
    """
    Tests the exact failing case from Issue #67.
    1000 * u.cm / u.s was evaluated as (1000 * u.cm) / u.s,
    and the bug in LinearQuantity.__truediv__(LinearUnit) caused an incorrect value.
    """
    cm =  u("cm")
    s =  u("s")

    # This is evaluated as (1000 * cm) / s
    q = 1000 * cm / s

    assert isinstance(q, LinearQuantity)
    assert q.unit.name == "cm/s"
    assert math.isclose(q._mag_si, 10.0)
    assert math.isclose(q.value, 1000.0)


@pytest.mark.regression(reason="Issue #67: Fix for LinearQuantity * LinearUnit constructor")
def test_regression_67_quantity_times_unit_uses_value():
    """
    Explicitly tests that (LinearQuantity) * (LinearUnit) uses self.value, not self._mag_si.
    """
    cm =  u("cm") # scale 0.01
    m =  u("m")   # scale 1.0

    q1 = 1000 * cm  # (value=1000, _mag_si=10.0)
    q2 = q1 * m     # retains composed unit name with prefix

    assert q2.unit.name == "cm^2"
    assert math.isclose(q2._mag_si, 10.0)
    assert math.isclose(q2.value, 100000.0)


@pytest.mark.regression(reason="Issue #67: Fix for LinearQuantity / LinearUnit constructor")
def test_regression_67_quantity_div_unit_uses_value():
    """
    Explicitly tests that (LinearQuantity) / (LinearUnit) uses self.value, not self._mag_si.
    """
    cm =  u("cm") # scale 0.01
    s =  u("s")   # scale 1.0

    q1 = 1000 * cm  # (value=1000, _mag_si=10.0)
    q2 = q1 / s

    assert q2.unit.name == "cm/s"
    assert math.isclose(q2._mag_si, 10.0)
    assert math.isclose(q2.value, 1000.0)


@pytest.mark.regression(reason="Bugfix: LinearQuantity * LinearUnit dimensionless path")
def test_quantity_times_unit_resulting_in_dimensionless():
    """
    Tests the `if new_unit.dim == DIM_0:` branch in LinearQuantity.__mul__.
    Ensures the SI magnitude is calculated correctly and a scale=1 unit is used.
    """
    m = u.m
    cm = u.cm

    # 1. Simple case: (10 m) * (1/m)
    q1 = 10 * m
    inv_m = 1 / m  # scale_to_si = 1.0

    q_final_1 = q1 * inv_m

    assert q_final_1.dim == DIM_0
    assert q_final_1.unit.scale_to_si == 1.0
    assert math.isclose(q_final_1._mag_si, 10.0)  # 10.0 * 1.0
    assert math.isclose(q_final_1.value, 10.0)

    # 2. Prefixed case: (10 m) * (1/cm)
    q2 = 10 * m  # _mag_si = 10.0
    inv_cm = 1 / cm  # scale_to_si = 1 / 0.01 = 100.0

    q_final_2 = q2 * inv_cm  # 10 m * (1 / 0.01 m) = 1000

    assert q_final_2.dim == DIM_0
    assert q_final_2.unit.scale_to_si == 1.0
    assert math.isclose(q_final_2._mag_si, 1000.0)  # 10.0 * 100.0
    assert math.isclose(q_final_2.value, 1000.0)

    # 3. Prefixed case (other way): (10 cm) * (1/m)
    q3 = 10 * cm  # _mag_si = 0.1
    # inv_m is from case 1 (scale_to_si = 1.0)

    q_final_3 = q3 * inv_m  # 10 cm * (1/m) = 0.1 m * (1/m) = 0.1

    assert q_final_3.dim == DIM_0
    assert q_final_3.unit.scale_to_si == 1.0
    assert math.isclose(q_final_3._mag_si, 0.1)  # 0.1 * 1.0
    assert math.isclose(q_final_3.value, 0.1)


@pytest.mark.regression(reason="Bugfix: Cover LinearQuantity * LinearUnit dimensionless path")
def test_quantity_times_inverse_unit_simple():
    """
    Explicitly tests the `if new_unit.dim == DIM_0:` branch in LinearQuantity.__mul__
    with a single simple case to satisfy code coverage.
    """
    m = u("m")
    q = 5 * m       # _mag_si = 5.0
    inv_m = 1 / m   # scale_to_si = 1.0

    # This calls q.__mul__(inv_m)
    result = q * inv_m
    
    # Test the exact lines from the coverage report
    # new_mag_si = self._mag_si * other.scale_to_si (5.0 * 1.0)
    # return LinearQuantity(new_mag_si, unit_dimless)
    assert result.dim == DIM_0
    assert result.unit.scale_to_si == 1.0
    assert math.isclose(result._mag_si, 5.0)
    assert math.isclose(result.value, 5.0)