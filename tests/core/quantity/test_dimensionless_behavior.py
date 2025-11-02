import pytest

from quantium.core.dimensions import DIM_0
from quantium.core.quantity import LinearQuantity
from quantium.units.registry import DEFAULT_REGISTRY as ureg


@pytest.mark.regression(reason="Issue #3: Verify all units produce dimensionless result when divided by themselves")
def test_all_units_self_division_is_dimensionless_and_nameless():
    """
    For every registered unit:
      - Create a LinearQuantity with magnitude 1 and that unit
      - Divide it by itself
      - Ensure the resulting dimension == DIM_0 (dimensionless)
      - Ensure the resulting unit name is "" (empty string)
      - Ensure __repr__ returns only the numeric value (no unit symbol)
    """
    for name, unit in ureg.all().items():
        q = LinearQuantity(1.0, unit)
        result = q / q

        # Check dimensionless
        assert result.dim == DIM_0, f"{name}: expected DIM_0, got {result.dim}"

        # Check unit name strictly empty string
        assert result.unit.name == "", f"{name}: expected unit name '', got '{result.unit.name}'"

        # Check __repr__ output: should be only a number (no space or unit)
        rep = repr(result)
        assert rep.strip().replace('.', '', 1).isdigit(), (
            f"{name}: expected numeric-only repr, got '{rep}'"
        )

@pytest.mark.regression(reason="Dimensionless results must be SI-normalized and numerically correct")
@pytest.mark.parametrize("sym_a, val_a, sym_b, val_b", [
    # base + prefixes
    ("m",   3.2,  "cm",  5.0),
    ("s",   7.0,  "ms",  2.0),
    ("g",   1.0,  "kg",  0.5),
    # derived units with prefixes
    ("N",   10.0, "uN",  5.0),
    ("Pa",  2.0,  "kPa", 3.0),
    ("J",   8.0,  "MJ",  2.0),
    ("Hz",  5.0,  "kHz", 1.0),
])
def test_dimensionless_result_is_si_and_correct(sym_a, val_a, sym_b, val_b):
    ua = ureg.get(sym_a)
    ub = ureg.get(sym_b)

    qa = val_a * ua
    qb = val_b * ub

    r = qa / qb

    # Expected pure number computed via SI magnitudes
    expected = (val_a * ua.scale_to_si) / (val_b * ub.scale_to_si)

    # 1) Dimensionless
    assert r.dim == DIM_0

    # 2) In SI (canonical dimensionless): empty name and scale 1.0
    assert r.unit.name == ""
    assert r.unit.scale_to_si == 1.0

    # 3) Numeric correctness
    # r is already in SI since r.unit.scale_to_si == 1.0
    assert float(repr(r)) == pytest.approx(expected)

