import pytest

from quantium.core.dimensions import DIM_0

from quantium.units.registry import DEFAULT_REGISTRY as ureg




# -------------------------------
# Reciprocal test for units
# -------------------------------
@pytest.mark.regression(reason="Issue #19: LinearUnit division with 1 error + edge cases")
def test_unit_reciprocal():
    s = ureg.get("s")

    # --- 1) No exceptions on valid reciprocal forms ---
    for expr in [lambda: 1/s, lambda: s**-1, lambda: 1/s**3, lambda: 1/s**-1]:
        try:
            expr()
        except Exception as e:
            pytest.fail(f"Reciprocal operation raised: {e}")

    # --- 2) Equivalence between reciprocal and power forms ---
    # Operator precedence note: ** binds tighter than /, so 1/s**-3 == 1/(s**-3)
    assert (1/s) == (s ** -1)
    assert (1/s**3) == (s ** -3)
    assert (1/s**-3) == (s ** 3)

    # --- 3) Idempotence / normalization of reciprocals ---
    # 1/(1/s) == s
    assert 1 / (1 / s) == s
    # Double-negative power via reciprocal should normalize:
    assert 1 / (s ** -1) == s
    # Reciprocal of a reciprocal stays stable (two flips -> original)
    assert 1 / (1 / (1 / s)) == (1 / s)

    # --- 4) Mixed powers with reciprocals ---
    # 1/(s^k) == s^-k for positive k
    assert 1 / (s ** 4) == (s ** -4)
    # 1/(s^-k) == s^k
    assert 1 / (s ** -5) == (s ** 5)

    # --- 5) Name normalization sanity (if names matter in equality) ---
    # Expect canonical power-style names (after your recent normalization)
    # These asserts fail only if names diverge while dim/scale match.
    assert (1 / s).name == (s ** -1).name
    assert (1 / (s ** 3)).name == (s ** -3).name
    assert (1 / (s ** -3)).name == (s ** 3).name

    # --- 6) Zero and one powers (corner cases) ---
    s0 = s ** 0               # dimensionless
    s1 = s ** 1
    # (1 / s^0) should flip exponent sign: ^0 -> ^-0 (still 0); stays dimensionless
    r0 = 1 / s0
    assert r0.dim == s0.dim
    assert r0.scale_to_si == pytest.approx(1.0)
    # 1/s^1 == s^-1
    assert 1 / s1 == (s ** -1)

    # --- 7) Chained expressions don't drift scale/dim ---
    # (1/s) * s -> dimensionless
    one = (1 / s) * s
    assert one.name  == '' # any name; just ensure object exists
    assert one.dim == DIM_0  # or however your dimensionless dim is represented

@pytest.mark.regression(reason="Issue #19: Invalid numerators must raise TypeError")
@pytest.mark.parametrize("n", [0, 2, 3, -1, 3.14])
def test_unit_reciprocal_invalid_numerators_raise(n):
    s = ureg.get("s")
    with pytest.raises(TypeError):
        _ = n / s

@pytest.mark.regression(reason="Issue #19: Parentheses / precedence and deep nesting")
def test_unit_reciprocal_parentheses_and_nesting():
    s = ureg.get("s")

    # Precedence: ** before /
    assert (1 / s**-1) == (1 / (s**-1)) == (s ** 1)

    # Deep nesting: 1/(1/(1/s)) == 1/s
    u = 1 / (1 / (1 / s))
    assert u == (1 / s)

    # Compound powers: ((1/s)**3) * (s**3) -> dimensionless
    left = (1 / s) ** 3
    right = s ** 3
    prod = left * right
    assert prod.dim == DIM_0
