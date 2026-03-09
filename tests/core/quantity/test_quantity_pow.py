import math

import pytest

from quantium.core.dimensions import DIM_0, LENGTH, TEMPERATURE, dim_div, dim_mul, dim_pow
from quantium.core.unit import LinearUnit
from quantium.units import u

def _name(sym: str, n: int) -> str:
    #return "1" if n == 0 else (sym if n == 1 else f"{sym}^{n}")
    if n == 0:
        return 1
    if n >= 1: 
        return f"{sym}^{n}"

    return f"1/{sym}^{abs(n)}"

def test_power_of_quantity():
    m = LinearUnit("m", 1.0, LENGTH)
    q2 = (2 * m) ** 2  # -> 4 m^2

    assert q2.dim == dim_pow(LENGTH, 2)
    assert q2.unit.name == "m^2"
    assert math.isclose(q2._mag_si / q2.unit.scale_to_si, 4.0)


@pytest.mark.regression(reason="Issue: #33 LinearUnit raised to power 0 is not dimensionless")
def test_quantity_pow_zero_and_one_and_negative():
    m = LinearUnit("m", 1.0, LENGTH)
    q = 2 * m
    q0 = q ** 0
    q1 = q ** 1
    qn = q ** -2
    assert q0.dim == DIM_0 and q0.unit.scale_to_si == 1.0
    assert q1.dim == LENGTH
    assert qn.dim == dim_pow(LENGTH, -2)


@pytest.mark.regression(reason="Issue #33: negative exponents should produce correct dim/name/scale")
@pytest.mark.parametrize("n", [-3, -4])
@pytest.mark.parametrize("sym, scale", [("m", 1.0), ("cm", 0.01)])
def test_unit_pow_negative_high_exponents_regression_issue_33(sym: str, scale: float, n: int):
    u = LinearUnit(sym, scale, LENGTH)
    up = u ** n

    # Dimension and name
    assert up.dim == dim_pow(LENGTH, n)
    assert up.name == _name(sym, n)

    # Scale: scale_to_si ** n (e.g., (0.01)**-3 = 1_000_000)
    assert up.scale_to_si == pytest.approx(scale ** n)

    # Reciprocal sanity
    assert (1 / (u ** (-n))) == (u ** n)


# -------------------------
# Regression: Issue #33 (LinearQuantity)
# -------------------------

@pytest.mark.regression(reason="Issue #33: LinearQuantity ** n must match dim, unit, and magnitude for negative exponents")
@pytest.mark.parametrize("n", [-3, -4])
@pytest.mark.parametrize("sym, scale, value", [
    ("m",  1.0,  2.0),
    ("cm", 0.01, 5.0),
])
def test_quantity_pow_negative_high_exponents_regression_issue_33(sym: str, scale: float, value: float, n: int):
    u = LinearUnit(sym, scale, LENGTH)
    q = value * u
    qp = q ** n

    # Dimension & unit name/scale
    assert qp.dim == dim_pow(LENGTH, n)
    assert qp.unit.name == _name(sym, n)
    assert qp.unit.scale_to_si == pytest.approx(scale ** n)

    # Magnitude in the resulting unit should be value**n
    mag_in_unit = qp._mag_si / qp.unit.scale_to_si
    assert math.isclose(mag_in_unit, value ** n, rel_tol=1e-12, abs_tol=1e-12)


def test_quantity_pow_with_fractions_repr():
    # Simple fractional power
    q = 100 * u.cm ** (1/3)
    q2 = 100 * u("cm^(1/3)")
    assert repr(q) == "100 cm^(1/3)"
    assert repr(q.to(u.m**(1/3))).endswith("m^(1/3)")
    assert repr(q.to('m^(1/3)')).endswith("m^(1/3)")
    assert q == q2

    # Joule fractional power
    q3 = 50 * u.J ** (1/2)
    q4 = 50 * u("J^(1/2)")
    assert repr(q3) == "50 J^(1/2)"
    assert repr(q3.to("kg^(1/2)*m/s")).endswith("kg^(1/2)·m/s")
    assert q3 == q4

    # Composite unit: Joule * meter
    q5 = 2 * (u.J * u.m) ** (1/3)
    q6 = 2 * u("J*m") ** (1/3)
    assert repr(q5) == "2 J^(1/3)·m^(1/3)"
    assert q5 == q6

    # Force unit: kg*m/s**2 fractional power
    q7 = 10 * (u.kg * u.m / u.s**2) ** (1/2)
    q8 = 10 * u("kg*m/s^2") ** (1/2)
    assert repr(q7) == "10 N^(1/2)"
    assert q7 == q8

    # Nested powers with mixed units
    q9 = 3 * (u.J**(2/3) * u.kg**(1/3))
    q10 = 3 * u("J^(2/3)*kg^(1/3)")
    assert repr(q9) == "3 J^(2/3)·kg^(1/3)"
    assert q9 == q10

@pytest.mark.parametrize("sym", [
    "m", "J", "N", "F", "kg*m/s**2", "Pa"
])
def test_caret_double_asterisk_return_same_power(sym : str):
    assert u(f'{sym}^(3/4)') == u(f'{sym}**(3/4)')
    assert u(f'{sym}^-11') == u(f'{sym}**-11')
    assert u(f'{sym}^-1') == u(f'{sym}**-1')
    assert u(f'{sym}^0') == u(f'{sym}**0')
    assert u(f'{sym}^3') == u(f'{sym}**3')
    assert u(f'{sym}^10') == u(f'{sym}**10')