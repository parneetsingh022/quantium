import pytest

from quantium.core.dimensions import DIM_0, LENGTH, dim_pow
from quantium.core.unit import LinearUnit

# NOTE:
# These tests rely on the classic binary rounding facts:
#   0.1 * 0.2 != 0.02 exactly
#   3 * 0.1 != 0.3 exactly
# so exact float equality will fail. The library should tolerate tiny drift.


@pytest.mark.regression(reason="Float drift: LinearUnit equality should tolerate tiny scale differences")
def test_unit_equality_tolerates_scale_float_drift():
    # Two mathematically identical scales obtained via different FP paths.
    u1 = LinearUnit("a", 0.1, LENGTH) * LinearUnit("b", 0.2, LENGTH)     # scale ~ 0.020000000000000004
    u2 = LinearUnit("c", 0.02, dim_pow(LENGTH, 2))                  # scale 0.02

    # If LinearUnit.__eq__ uses exact float equality, this will fail.
    # After fixing, __eq__ should consider them equal (same dim, scales within tolerance).
    assert u1 == u2, f"scales differ slightly: {u1.scale_to_si} vs {u2.scale_to_si}"


@pytest.mark.regression(reason="Float drift: LinearQuantity equality should use tolerant SI magnitude comparison")
def test_quantity_equality_tolerates_si_float_drift():
    m = LinearUnit("m", 1.0, LENGTH)
    a = LinearUnit("a", 0.1, LENGTH)

    q1 = 3 * a      # _mag_si ~ 0.30000000000000004
    q2 = 0.3 * m    # _mag_si ~ 0.29999999999999999

    # Exact equality on _mag_si will fail. After fix, should pass.
    assert q1 == q2, f"_mag_si differ slightly: {q1._mag_si} vs {q2._mag_si}"


@pytest.mark.regression(reason="Float drift: Dimensionless ratios should be numerically 1 within tolerance")
def test_dimensionless_ratio_avoids_float_drift():
    m = LinearUnit("m", 1.0, LENGTH)
    a = LinearUnit("a", 0.1, LENGTH)

    num = 3 * a       # SI ~ 0.30000000000000004
    den = 0.3 * m     # SI ~ 0.29999999999999999
    r = num / den     # should be dimensionless 1

    # 1) Dimensionless dim
    assert r.dim == DIM_0

    # 2) Canonical dimensionless unit (empty name, scale 1.0)
    assert r.unit.name == ""
    assert r.unit.scale_to_si == 1.0

    # 3) Numeric value ~ 1, allow tiny drift (repr shows the magnitude in current unit)
    # If repr prints a bare number for dimensionless quantities, cast to float safely:
    val = float(repr(r))
    assert val == pytest.approx(1.0, rel=1e-12, abs=1e-15)