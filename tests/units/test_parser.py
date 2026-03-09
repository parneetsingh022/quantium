# tests/test_parser.py
import math
import pytest

from quantium.units.parser import (
    _UnitExprParser,
    _compile_unit_expr,
    extract_unit_expr,
)
from quantium.units.registry import UnitsRegistry, DEFAULT_REGISTRY
from quantium.core.dimensions import (
    LENGTH, TIME, MASS, CURRENT, DIM_0, dim_div, dim_mul, dim_pow
)
from quantium.core.quantity import LinearQuantity
from quantium.core.unit import LinearUnit


# --------------------------
# Parsing-only unit tests
# --------------------------

def test_parse_simple_name():
    plan = _UnitExprParser("m").parse()
    assert plan[0] == "name" and plan[1] == "m"

def test_parse_mul_div_and_precedence_smoke():
    plan = _UnitExprParser("kg*m/s**2").parse()
    # shape: (((kg * m) / (s**2))) left-assoc for * and /
    assert plan[0] in {"div", "mul"}

def test_parse_parentheses_and_pow():
    plan = _UnitExprParser("(m/s)**2").parse()
    assert plan[0] == "pow" and plan[2] == 2

def test_parse_signed_exponents_positive():
    plan = _UnitExprParser("m**+3").parse()
    assert plan[0] == "pow" and plan[2] == 3

def test_parse_signed_exponents_negative():
    plan = _UnitExprParser("s**-2").parse()
    assert plan[0] == "pow" and plan[2] == -2

def test_parse_ignores_whitespace():
    plan = _UnitExprParser("  kg *  m  /  s ** 2 ").parse()
    assert plan  # parsed successfully

def test_parse_unbalanced_parenthesis_raises():
    with pytest.raises(ValueError):
        _UnitExprParser("(m").parse()

def test_parse_missing_exponent_raises():
    with pytest.raises(ValueError):
        _UnitExprParser("m**").parse()

def test_prefilter_disallowed_characters():
    with pytest.raises(ValueError):
        _compile_unit_expr("m+s")  # '+' not allowed except as exponent sign
    with pytest.raises(ValueError):
        _compile_unit_expr("m[2]")

def test_trailing_garbage_raises():
    with pytest.raises(ValueError):
        _UnitExprParser("m)").parse()

def test_name_with_digit_tail_is_single_token_not_literal_one():
    # Grammar allows digits after first char; "m1" is a NAME token
    plan = _UnitExprParser("m1").parse()
    assert plan == ("name", "m1", None)


# --------------------------
# Evaluation using real registry
# --------------------------

def test_eval_simple_name_with_default_registry():
    reg = DEFAULT_REGISTRY
    u = extract_unit_expr("m", reg)
    assert isinstance(u, LinearUnit)
    assert u.dim == LENGTH
    assert math.isclose(u.scale_to_si, 1.0)

def test_eval_mul_div_pow_chain_dims_and_scale():
    reg = DEFAULT_REGISTRY
    u = extract_unit_expr("kg*m/s**2", reg)  # should be N (SI)
    assert u.dim == dim_div(dim_mul(MASS, LENGTH), dim_pow(TIME, 2))
    assert math.isclose(u.scale_to_si, 1.0)

def test_eval_parenthesized_pow():
    reg = DEFAULT_REGISTRY
    u = extract_unit_expr("(m/s)**2", reg)
    assert u.dim == dim_div(dim_pow(LENGTH, 2), dim_pow(TIME, 2))
    assert math.isclose(u.scale_to_si, 1.0)

def test_eval_left_associative_division():
    # a/b/c == (a/b)/c; use custom registry with atomic 'a','b','c'
    reg = UnitsRegistry()
    reg.register(LinearUnit("a", 1.0, LENGTH))
    reg.register(LinearUnit("b", 2.0, TIME))
    reg.register(LinearUnit("c", 5.0, MASS))
    u = extract_unit_expr("a/b/c", reg)
    assert u.dim == dim_div(dim_div(LENGTH, TIME), MASS)
    assert math.isclose(u.scale_to_si, 1.0 / (2.0 * 5.0))

def test_eval_literal_one_identity():
    u = extract_unit_expr("1", DEFAULT_REGISTRY)
    assert u.dim == DIM_0 and math.isclose(u.scale_to_si, 1.0)

def test_eval_mul_with_one_does_not_change():
    u = extract_unit_expr("m*1/s", DEFAULT_REGISTRY)
    assert u.dim == dim_div(LENGTH, TIME)
    assert math.isclose(u.scale_to_si, 1.0)

def test_eval_signed_exponent_negative_scale_and_dims():
    u = extract_unit_expr("s**-3", DEFAULT_REGISTRY)
    assert u.dim == dim_pow(TIME, -3)
    assert math.isclose(u.scale_to_si, 1.0)

def test_eval_large_positive_and_negative_exponents():
    reg = UnitsRegistry()
    reg.register(LinearUnit("x", 10.0, LENGTH))
    reg.register(LinearUnit("y", 0.5, TIME))
    u = extract_unit_expr("x**10 / y**-7", reg)
    # x^10 * y^7
    assert u.dim == dim_mul(dim_pow(LENGTH, 10), dim_pow(TIME, 7))
    assert math.isclose(u.scale_to_si, (10.0 ** 10) * ((0.5) ** 7))


# --------------------------
# Caching behavior (plan only)
# --------------------------

def test_compile_cache_reuses_plan_but_evaluates_against_each_registry():
    # Two different registries with different factor for 'm'
    reg1 = UnitsRegistry()
    reg2 = UnitsRegistry()
    reg1.register(LinearUnit("m", 1.0, LENGTH))
    reg2.register(LinearUnit("m", 2.0, LENGTH))

    # Plan is cached by expression string
    p1 = _compile_unit_expr("m")
    p2 = _compile_unit_expr("m")
    assert p1 is p2

    u1 = extract_unit_expr("m", reg1)
    u2 = extract_unit_expr("m", reg2)
    assert math.isclose(u1.scale_to_si, 1.0)
    assert math.isclose(u2.scale_to_si, 2.0)


# --------------------------
# Registry integration extras
# --------------------------

def test_registry_delegates_to_parser_for_composed_expressions():
    u = DEFAULT_REGISTRY.get("m/s")
    assert u.dim == dim_div(LENGTH, TIME)
    assert math.isclose(u.scale_to_si, 1.0)

def test_prefix_synthesis_kilo_and_micro():
    kW = DEFAULT_REGISTRY.get("kW")
    assert kW.dim == dim_div(dim_mul(MASS, dim_pow(LENGTH, 2)), dim_pow(TIME, 3))  # W dim
    assert math.isclose(kW.scale_to_si, 1000.0)

    uF = DEFAULT_REGISTRY.get("uF")  # ASCII 'u' → µ normalization
    # Farad is dim of capacitance; factor should be 1e-6
    assert math.isclose(uF.scale_to_si, 1e-6)

def test_alias_ohm_variants_map_to_omega():
    for alias in ("ohm", "Ohm", "OHM"):
        r = DEFAULT_REGISTRY.get(alias)
        omega = DEFAULT_REGISTRY.get("Ω")
        assert r == omega

def test_non_prefixable_units_reject_prefix():
    with pytest.raises(ValueError):
        DEFAULT_REGISTRY.get("kmin")  # minutes are non-prefixable

def test_quantity_and_unit_roundtrip_repr_and_conversion():
    # 1000 cm/s should print as "1000 cm/s" by default, and "10 m/s" in SI
    cm = LinearUnit("cm", 0.01, LENGTH)
    s = DEFAULT_REGISTRY.get("s")
    v = LinearQuantity(1000, cm / s)
    assert f"{v}" == "1000 cm/s"
    assert f"{v:si}" == "10 m/s"

def test_unit_name_collapse_on_same_unit_multiplication():
    s = DEFAULT_REGISTRY.get("s")
    ss = s * s
    # name normalization to power is expected (e.g., "s^2")
    assert "^2" in ss.name
    assert ss.dim == dim_pow(TIME, 2)

def test_one_over_unit_name_formatting():
    s = DEFAULT_REGISTRY.get("s")
    inv = LinearUnit("1", 1.0, DIM_0) / s   # via LinearUnit.__truediv__
    assert inv.name == "1/s"  # depending on your formatting path
    assert inv.dim == dim_pow(TIME, -1)
