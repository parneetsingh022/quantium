import math
import pytest
import operator


from quantium.core.dimensions import LENGTH, TIME, TEMPERATURE, DIM_0
from quantium.core.quantity import LinearQuantity
from quantium.core.unit import LinearUnit
from quantium.units.registry import DEFAULT_REGISTRY as dreg
from quantium.units import u
# -------------------------------
# LinearQuantity: basics & conversion
# -------------------------------

def test_quantity_construct_and_to():
    m  = LinearUnit("m", 1.0, LENGTH)
    cm = LinearUnit("cm", 0.01, LENGTH)

    q_cm = LinearQuantity(200, cm)          # 200 cm
    q_m  = q_cm.to(m)                  # -> 2 m

    assert isinstance(q_m, LinearQuantity)
    assert q_m.unit == m
    assert q_m.dim == LENGTH
    # _mag_si is internal, so check using units:
    assert math.isclose(q_m._mag_si, 2.0)  # 2 m in SI
    # magnitude shown in the *current* unit:
    assert math.isclose(q_m._mag_si / q_m.unit.scale_to_si, 2.0)

def test_quantity_to_dimension_mismatch_raises():
    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TIME)
    q = LinearQuantity(3, m)
    with pytest.raises(TypeError):
        q.to(s)


# -------------------------------
# __rmatmul__: value * LinearUnit
# -------------------------------

def test_rmatmul_operator():
    m = LinearUnit("m", 1.0, LENGTH)
    q = 3 * m
    assert isinstance(q, LinearQuantity)
    assert q.dim == LENGTH
    assert q.unit is m
    assert math.isclose(q._mag_si, 3.0)



# ----------------------------
# Helpers
# ----------------------------
def shown(q: LinearQuantity) -> float:
    """Return the magnitude shown in q's current unit (not SI)."""
    return q._mag_si / q.unit.scale_to_si


# ----------------------------
# Happy-path conversions using string expressions
# ----------------------------

def test_to_string_simple_prefix_change_velocity():
    # 10 m/s -> 1000 cm/s
    q = 10 * dreg.get("m/s")
    out = q.to("cm/s")
    assert math.isclose(shown(out), 1000.0)
    assert out.dim == q.dim

def test_to_string_acceleration_power_syntax():
    # 9.8 m/s^2 -> 980 cm/s^2, using ** syntax required by your parser
    q = 9.8 * dreg.get("m/s**2")
    out = q.to("cm/s**2")
    assert math.isclose(shown(out), 980.0, rel_tol=1e-12)

def test_to_string_force_from_newton_to_base_composed():
    # 1 N -> 1 kg·m/s^2
    q = 1 * dreg.get("N")
    out = q.to("kg*m/s**2")
    assert math.isclose(shown(out), 1.0)
    assert out.dim == q.dim

def test_to_string_energy_newton_meter_equivalence():
    # 3 J -> 3 N·m
    q = 3 * dreg.get("J")
    out = q.to("N*m")
    assert math.isclose(shown(out), 3.0)
    assert out.dim == q.dim

def test_to_string_power_joule_per_second():
    # 7 W -> 7 J/s
    q = 7 * dreg.get("W")
    out = q.to("J/s")
    assert math.isclose(shown(out), 7.0)
    assert out.dim == q.dim

def test_to_string_frequency_from_kHz_to_one_over_s():
    # 1 kHz -> 1000 1/s
    q = 1 * dreg.get("kHz")
    out = q.to("1/s")
    assert math.isclose(shown(out), 1000.0)
    assert out.dim == q.dim

def test_to_string_pressure_pascal_to_N_per_m2():
    # 101325 Pa -> 101325 N/m^2
    q = 101_325 * dreg.get("Pa")
    out = q.to("N/m**2")
    assert math.isclose(shown(out), 101_325.0)
    assert out.dim == q.dim

def test_to_string_parentheses_and_mixed_ops():
    # 100 m/s -> 10000 (cm)/s using parentheses and explicit *
    q = 100 * dreg.get("m/s")
    out = q.to("(cm) / (s)")
    assert math.isclose(shown(out), 10_000.0)
    assert out.dim == q.dim

def test_to_string_with_micro_alias_in_denominator():
    # 1 / ms -> 1000 1/s (Hz dimension), using 'ms' in target string
    q = 1 * (1 / dreg.get("ms"))  # LinearQuantity with T^-1
    out = q.to("1/s")
    assert math.isclose(shown(out), 1000.0)
    assert out.dim == q.dim

def test_to_string_with_ohm_alias_normalization():
    # 'ohm' alias should resolve to 'Ω' under the hood
    q = 5 * dreg.get("Ω")
    out = q.to("ohm")
    assert math.isclose(shown(out), 5.0)
    # Registry normalizes alias to canonical "Ω"
    assert out.unit.name == "Ω"
    assert out.dim == q.dim


# --------------------------------------------------------
# Tests conversion from one naming system to other
# --------------------------------------------------------

def test_to_physically_equivalent_different_name():
    """
    Tests the bug fix: converting to a unit that is physically
    identical but has a different name should return a NEW object.
    """
    q1 = LinearQuantity(5.0, u.W/(u.A*u.m))  # 5.0 W/(A·m)

    # Test conversion using a LinearUnit object
    q2 = q1.to(u.V/u.m)            # Convert to V/m

    # 1. Check physical equivalence (value is the same)
    assert q1 == q2
    assert f"{q2}" == "5 V/m"

    # 2. Check that it is a NEW object with the new unit name
    assert q1 is not q2
    assert q2.unit.name == 'V/m'
    assert q1.unit.name == 'W/(A·m)' # Original is unchanged

    # Test conversion using a string name
    q3 = q1.to("V/m")            # Convert to V/m

    # 3. Check physical equivalence
    assert q1 == q3
    assert f"{q3}" == "5 V/m"

    # 4. Check that it is a NEW object
    assert q1 is not q3
    assert q3.unit.name == 'V/m'

def test_to_identical_name_optimization():
    """
    Tests the optimization path: converting to the *exact same unit*
    (identical name) should return the SAME object (`self`).
    """
    q1 = LinearQuantity(10.0, u.V/u.m)  # 10.0 V/m

    # Test conversion using the *same* LinearUnit object
    q2 = q1.to(u.V/u.m)

    # Check that it returned the *exact same object*
    assert q1 is q2
    assert f"{q2}" == "10 V/m"

    # Test conversion using the *same* string name
    q3 = q1.to("V/m")

    # Check that this also returned the *exact same object*
    assert q1 is q3
    assert f"{q3}" == "10 V/m"

# ----------------------------
# Identity / fast path
# ----------------------------

def test_to_string_identity_fast_path_returns_same_object():
    # .to("m") on a LinearQuantity already in meters should return self
    q = 2.5 * dreg.get("m")
    r = q.to("m")
    assert r is q


# ----------------------------
# Dimension mismatch and invalid syntax
# ----------------------------

def test_to_string_dimension_mismatch_raises():
    # 3 m -> "kg" should raise TypeError
    q = 3 * dreg.get("m")
    with pytest.raises(TypeError):
        _ = q.to("kg")

def test_to_string_invalid_expression_raises_value_error():
    # Parser should reject invalid tokens like '//' and raise ValueError
    q = 1 * dreg.get("m/s")
    with pytest.raises(ValueError):
        _ = q.to("m//s")


# ----------------------------
# More complex target expressions
# ----------------------------

def test_to_string_complex_nested_equivalence():
    # 1 N -> (kg·m)/(s^2) using parentheses and ** syntax
    q = 1 * dreg.get("N")
    out = q.to("(kg*m)/(s**2)")
    assert math.isclose(shown(out), 1.0)
    assert out.dim == q.dim

def test_to_string_uses_registry_parser_consistently_with_sources():
    # Construct a unit via a complex expression and convert to a different but equivalent one
    q = 4 * dreg.get("(W*s)/(N*s/m**2)")  # dimension L^3/T
    out = q.to("m**3/s")
    assert math.isclose(shown(out), 4.0)
    assert out.dim == q.dim


# ----------------------------
# Dimensionless support
# ----------------------------

def test_to_string_dimensionless_no_change():
    # (10 s) / (2 s) -> 5 (dimensionless); converting to "1" keeps it dimensionless
    s = dreg.get("s")
    q = (10 * s) / (2 * s)
    assert q.dim == DIM_0
    out = q.to("1")
    assert out.dim == DIM_0
    assert math.isclose(shown(out), 5.0)

# ----------------------------
# VALUE PROPERTY
# ----------------------------

def test_quantity_value_property():
    """Tests the new .value property."""
    
    # 1. Simple quantity in base unit
    q_m = 3 * u.m
    assert isinstance(q_m.value, float)
    assert math.isclose(q_m.value, 3.0)

    # 2. Simple quantity in non-base unit
    q_cm = 200 * u.cm
    assert math.isclose(q_cm.value, 200.0)

    # 3. LinearQuantity after conversion
    # 200 cm -> 2 m
    q_m_converted = q_cm.to(u.m)
    assert math.isclose(q_m_converted.value, 2.0)
    
    # 4. Check that the original quantity's value is unchanged
    assert math.isclose(q_cm.value, 200.0)

    # 5. Complex quantity
    q_accel = 9.8 * (u.m / u.s**2)
    assert math.isclose(q_accel.value, 9.8)

    # 6. Complex quantity after conversion
    # 9.8 m/s^2 -> 980 cm/s^2
    q_accel_cm = q_accel.to(u.cm / u.s**2)
    assert math.isclose(q_accel_cm.value, 980.0)

    # 7. Dimensionless quantity
    q_dimless = (10 * u.m) / (5 * u.m)
    assert q_dimless.dim == DIM_0
    assert math.isclose(q_dimless.value, 2.0)
    
    # 8. Sanity check against your 'shown' helper
    assert math.isclose(q_accel_cm.value, shown(q_accel_cm))

    # 9. Test (int + Dimension)
    # This fails because int.__add__ fails, and Dimension.__radd__
    # correctly returns NotImplemented.
    with pytest.raises(TypeError, match="unsupported operand type"):
        _ = 1 + LENGTH


# defensive guard when parser doesn't return a LinearUnit ---
def test_to_parser_returns_non_linearunit_triggers_guard(monkeypatch):
    """
    If the unit parser returns something that is NOT a LinearUnit
    (e.g., future AffineUnit or a bug), .to(...) should raise the
    defensive TypeError.
    """
    import quantium.core.quantity as qmod  # module under test

    class NotAUnit:
        pass

    # Patch the *bound* name used by LinearQuantity.to(...)
    monkeypatch.setattr(qmod, "extract_unit_expr", lambda s, reg: NotAUnit())

    q = 1 * dreg.get("m")
    with pytest.raises(TypeError, match="did not resolve to a LinearUnit"):
        _ = q.to("anything")  # string won’t be parsed; our patch returns NotAUnit


# ----------------------------
# _check_dim_compatible(): compare to 0
# ----------------------------

def test_compare_dimensioned_quantity_to_zero_raises_typeerror():
    q = 3 * u.m  # dimensioned
    for op in (operator.lt, operator.le, operator.gt, operator.ge):
        with pytest.raises(TypeError, match="Cannot compare a dimensioned quantity to 0"):
            _ = op(q, 0)  # q < 0, q <= 0, q > 0, q >= 0

def test_compare_dimensionless_quantity_to_zero_is_allowed():
    q = (10 * u.s) / (5 * u.s)  # dimensionless (== 2)
    assert q.dim == DIM_0

    # Should NOT raise; comparisons should behave numerically vs 0
    assert (q > 0) is True
    assert (q >= 0) is True
    assert (q < 0) is False
    assert (q <= 0) is False

# ----------------------------
# _check_dim_compatible(): wrong-type operand
# ----------------------------

def test_compare_with_non_quantity_non_number_raises_typeerror():
    q = 1 * u.m
    # Use a comparison that triggers _check_dim_compatible (not __eq__)
    with pytest.raises(TypeError) as excinfo:
        _ = q < "oops"
    # Message should include the offending type
    assert "Cannot compare LinearQuantity with type" in str(excinfo.value)
    assert "str" in str(excinfo.value)


# ----------------------------
# Delta temperature units: Δ°C, Δ°F, Δ°R
# ----------------------------

def test_delta_c_to_delta_kelvin():
    # 10 Δ°C == 10 K (no offset for deltas)
    q = 10 * u("Δ°C")
    out = q.to("delta_k")
    assert math.isclose(shown(out), 10.0)
    assert out.dim == q.dim

def test_delta_f_to_delta_kelvin():
    # 18 Δ°F == 10 K  (scale 5/9)
    q = 18 * u("Δ°F")
    out = q.to("delta_K")
    assert math.isclose(shown(out), 10.0, rel_tol=1e-12)
    assert out.dim == q.dim

def test_delta_r_to_delta_kelvin():
    # 9 Δ°R == 5 K  (scale 5/9)
    q = 9 * u("Δ°R")
    out = q.to("delta_K")
    assert math.isclose(shown(out), 5.0, rel_tol=1e-12)
    assert out.dim == q.dim

def test_delta_c_to_delta_f():
    # 25 Δ°C == 45 Δ°F
    q = 25 * u("Δ°C")
    out = q.to("Δ°F")
    assert math.isclose(shown(out), 45.0, rel_tol=1e-12)
    assert out.dim == q.dim
    assert out.unit.name == "Δ°F"

def test_delta_f_to_delta_c_roundtrip_precision():
    # 123 Δ°F -> Δ°C -> Δ°F should match (within tight tolerance)
    q = 123 * u("Δ°F")
    c = q.to("Δ°C")
    f = c.to("Δ°F")
    assert math.isclose(shown(f), 123.0, rel_tol=1e-12, abs_tol=0.0)

def test_delta_r_to_delta_c_and_f():
    # 18 Δ°R == 10 Δ°C == 18 Δ°R; also Δ°F should be 18 * (5/9)*9/5 == 10 Δ°F? Nope.
    # Better: 10 Δ°C == 18 Δ°R and 10 Δ°C == 18 Δ°F /? Wait:
    # Relationship: 1 Δ°C = 1 K; 1 Δ°F = 5/9 K; 1 Δ°R = 5/9 K.
    # So 18 Δ°R -> K = 18*(5/9)=10 K -> Δ°C = 10; -> Δ°F = 10*(9/5)=18.
    q = 18 * u("Δ°R")
    to_c = q.to("Δ°C")
    to_f = q.to("Δ°F")
    assert math.isclose(shown(to_c), 10.0, rel_tol=1e-12)
    assert math.isclose(shown(to_f), 18.0, rel_tol=1e-12)

def test_delta_addition_across_units_uses_si_and_returns_left_unit():
    # 5 Δ°C + 9 Δ°R: 9 Δ°R = 9*(5/9)=5 K; 5 Δ°C = 5 K → sum = 10 K
    # Should return in the left operand's unit (Δ°C) with value 10.
    q1 = 5 * u("Δ°C")
    q2 = 9 * u("Δ°R")
    s = q1 + q2
    assert s.unit.name == "Δ°C"
    assert math.isclose(shown(s), 10.0, rel_tol=1e-12)

def test_delta_string_conversion_parsing_symbols():
    # Ensure the parser resolves the unicode “Δ” names
    q = 12.5 * u("Δ°F")
    out = q.to("Δ°C")
    # 12.5 Δ°F -> K = 12.5*(5/9)= 6.944444..., so Δ°C = same numeric as K
    assert math.isclose(shown(out), 12.5 * (5/9), rel_tol=1e-12)

def test_delta_units_keep_temperature_dimension():
    for sym in ("Δ°C", "Δ°F", "Δ°R"):
        q = 1 * u(sym)
        assert q.dim == TEMPERATURE