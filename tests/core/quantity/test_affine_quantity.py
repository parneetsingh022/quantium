import pytest
from math import isclose

from quantium.core.dimensions import TEMPERATURE
from quantium.core.unit import LinearUnit
from quantium.core.quantity import LinearQuantity, AffineQuantity
from quantium.core.unit import AffineUnit
from quantium.units import u

# --- Fixtures: temperature units ---

@pytest.fixture
def K():
    # Kelvin (linear, delta and absolute are the same scale 1.0)
    return u.K

@pytest.fixture
def dK():
    # Kelvin (linear, delta and absolute are the same scale 1.0)
    return u.delta_k

@pytest.fixture
def dC():
    # Δ°C is linear with scale 1 to K for deltas
    return u.delta_degC

@pytest.fixture
def dF():
    # Δ°F: 1 °F difference = 5/9 K
    return u.delta_degF

@pytest.fixture
def degC():
    # °C: T(K) = T(°C) + 273.15
    return u.degC

@pytest.fixture
def degF():
    # °F: T(K) = (T(°F) - 32) * 5/9 + 273.15 = (5/9)*T(°F) + 255.372222...
    return u.degF

@pytest.fixture
def degR():
    # Rankine: T(K) = (5/9) * T(°R)
    return u.degR


# --- Core construction & conversions ---

def test_scalar_times_affine_unit_builds_affine_quantity(degC):
    t = 100 * degC
    assert isinstance(t, AffineQuantity)
    assert t.unit is degC
    # 100 °C = 373.15 K
    assert isclose(t._mag_si, 373.15, rel_tol=0, abs_tol=1e-12)
    assert isclose(t.value, 100.0, rel_tol=0, abs_tol=1e-12)

def test_affine_to_affine_conversion(degC, degF):
    # 100 °C == 212 °F
    tC = 100 * degC
    tF = tC.to(degF)
    assert isinstance(tF, AffineQuantity)
    assert tF.unit is degF
    assert isclose(tF.value, 212.0, rel_tol=0, abs_tol=1e-12)
    # round-trip back to °C
    tC2 = tF.to(degC)
    assert isclose(tC2.value, 100.0, rel_tol=0, abs_tol=1e-12)

def test_to_si_returns_affine_in_si_family_symbol_when_possible(degC):
    # Expect SI-style affine (scale=1, offset=0) with symbol "K" if preferred_symbol_for_dim -> "K"
    t = 25 * degC  # 298.15 K internally
    t_si = t.to_si()
    assert isinstance(t_si, AffineQuantity)
    assert isclose(t_si._mag_si, 298.15, rel_tol=0, abs_tol=1e-12)
    # value in its own unit (offset=0, scale=1) equals SI magnitude
    assert isclose(t_si.value, 298.15, rel_tol=0, abs_tol=1e-12)
    # unit name may be "K" depending on your preferred_symbol_for_dim; don't hard assert the name


# --- Arithmetic semantics ---

def test_affine_minus_affine_is_delta_in_unit_delta_unit(degC):
    t1 = 100 * degC  # 373.15 K
    t2 = 20 * degC   # 293.15 K
    d = t1 - t2
    assert isinstance(d, LinearQuantity)
    assert d.unit.name == "Δ°C"  # as provided via degC.delta_unit
    # 80 K difference
    assert isclose(d._mag_si, 80.0, rel_tol=0, abs_tol=1e-12)
    # Check representation value in Δ°C (scale=1)
    assert isclose(d.value, 80.0, rel_tol=0, abs_tol=1e-12)

def test_affine_plus_delta_is_affine(degC, dK):
    t = 10 * degC   # 283.15 K
    d = 5 * dK       # 5 K delta (LinearQuantity)
    out = t + d
    assert isinstance(out, AffineQuantity)
    # 288.15 K -> 15 °C
    assert isclose(out._mag_si, 288.15, rel_tol=0, abs_tol=1e-12)
    assert isclose(out.value, 15.0, rel_tol=0, abs_tol=1e-12)

def test_affine_minus_delta_is_affine(degC, dK):
    t = 10 * degC   # 283.15 K
    d = 3 * dK       # 3 K delta
    out = t - d
    assert isinstance(out, AffineQuantity)
    # 280.15 K -> 7 °C
    assert isclose(out._mag_si, 280.15, rel_tol=0, abs_tol=1e-12)
    assert isclose(out.value, 7.0, rel_tol=0, abs_tol=1e-12)

def test_affine_add_affine_raises(degC):
    with pytest.raises(TypeError):
        _ = (10 * degC) + (5 * degC)

def test_delta_unit_for_fahrenheit_differences(degF):
    t1 = 212 * degF  # = 373.15 K
    t2 = 32 * degF   # = 273.15 K
    d = t1 - t2
    # difference is 180 °F -> 100 K; delta unit should be Δ°F with scale 5/9
    assert isinstance(d, LinearQuantity)
    assert d.unit.name == "Δ°F"
    assert isclose(d._mag_si, 100.0, rel_tol=0, abs_tol=1e-12)
    # value in Δ°F should be 180 (since 180 * 5/9 = 100 K)
    assert isclose(d.value, 180.0, rel_tol=0, abs_tol=1e-12)

def test_rankine_behaves_as_expected_for_absolute(degR):
    # 671.67 °R == 373.15 K (boiling)
    tR = 671.67 * degR
    assert isclose(tR._mag_si, 373.15, rel_tol=0, abs_tol=5e-8)


# --- Comparisons ---

def test_affine_comparisons(degC):
    t1 = 25 * degC     # 298.15 K
    t2 = 24.999999999999 * degC
    # equality uses isclose internally
    assert t1 == t2 or (t1 >= t2 and t1 <= t2)
    # ordering against a clearly smaller one
    t3 = 20 * degC     # 293.15 K
    assert t1 > t3
    assert t3 < t1
    assert t1 >= t3
    assert t3 <= t1


# --- Formatting & hashing ---

def test_repr_and_format(degC):
    t = 12.34 * degC
    r = repr(t)
    assert "°C" in r
    assert "12.34" in r
    # format spec "si" should print SI-view
    s = f"{t:si}"
    assert "12.34" not in s  # different numeric value
    assert "°C" not in s     # unit likely switches to K family

def test_as_key_hashing_stability(degC):
    t1 = (1.0 + 1e-14) * degC
    t2 = (1.0 - 1e-14) * degC
    k1 = t1.as_key(precision=12)
    k2 = t2.as_key(precision=12)
    assert k1 == k2


# --- Invalid operations ---

def test_scaling_affine_scalar_raises(degC):
    with pytest.raises(TypeError):
        _ = (10 * degC) * 2
    with pytest.raises(TypeError):
        _ = 2 * (10 * degC)

def test_affine_mul_div_with_units_or_quantities_raise(degC, K):
    t = 10 * degC
    with pytest.raises(TypeError):
        _ = t * K
    with pytest.raises(TypeError):
        _ = t / K
    with pytest.raises(TypeError):
        _ = t * (1 * K)
    with pytest.raises(TypeError):
        _ = (1 * K) * t
    with pytest.raises(TypeError):
        _ = (1 * K) / t
    with pytest.raises(TypeError):
        _ = t ** 2


def test_affine_minus_affine_different_units(degC, degF):
    """
    Tests Point - Point subtraction with different units.
    The resulting delta's unit should match the delta_unit of the LEFT operand.
    """
    tC = 100 * degC   # 373.15 K
    tF = 32 * degF    # 273.15 K
    
    # tC - tF = 100 K difference. Result unit should be Δ°C (from tC).
    # A 100 K difference is 100 Δ°C.
    d_C = tC - tF
    assert isinstance(d_C, LinearQuantity)
    assert d_C.unit.name == "Δ°C"
    assert isclose(d_C._mag_si, 100.0, rel_tol=0, abs_tol=1e-12)
    assert isclose(d_C.value, 100.0, rel_tol=0, abs_tol=1e-12)

    # tF - tC = -100 K difference. Result unit should be Δ°F (from tF).
    # A -100 K difference is -180 Δ°F.
    d_F = tF - tC
    assert isinstance(d_F, LinearQuantity)
    assert d_F.unit.name == "Δ°F"
    assert isclose(d_F._mag_si, -100.0, rel_tol=0, abs_tol=1e-12)
    assert isclose(d_F.value, -180.0, rel_tol=0, abs_tol=1e-12)

def test_affine_plus_delta_different_units(degC, dF):
    """Tests Point + Vector where the Vector's unit is different."""
    tC = 10 * degC   # 283.15 K
    dF = 18 * dF   # 18 * (5/9) = 10 K delta
    
    # Result should be 293.15 K, expressed in the Point's unit (°C)
    # 293.15 K = 20 °C
    t_new = tC + dF
    assert isinstance(t_new, AffineQuantity)
    assert t_new.unit is degC  # Stays in the original affine unit
    assert isclose(t_new._mag_si, 293.15, rel_tol=0, abs_tol=1e-12)
    assert isclose(t_new.value, 20.0, rel_tol=0, abs_tol=1e-12)

def test_affine_minus_delta_different_units(degF, dC):
    """Tests Point - Vector where the Vector's unit is different."""
    tF = 50 * degF   # 283.15 K
    dC = 10 * dC   # 10 K delta (same as 10 * K)
    
    # Result should be 273.15 K, expressed in the Point's unit (°F)
    # 273.15 K = 32 °F
    t_new = tF - dC
    assert isinstance(t_new, AffineQuantity)
    assert t_new.unit is degF # Stays in the original affine unit
    assert isclose(t_new._mag_si, 273.15, rel_tol=0, abs_tol=1e-12)
    assert isclose(t_new.value, 32.0, rel_tol=0, abs_tol=1e-12)


def test_affine_comparisons_different_units(degC, degF):
    """Tests comparisons (==, >, <) between different affine units."""
    tC_boil = 100 * degC  # 373.15 K
    tF_boil = 212 * degF  # 373.15 K
    
    tF_hotter = 213 * degF # 373.705... K
    tF_colder = 211 * degF # 372.594... K

    # Equality
    assert tC_boil == tF_boil
    assert not (tC_boil != tF_boil)

    # Greater/Less than
    assert tC_boil < tF_hotter
    assert tF_hotter > tC_boil
    assert tC_boil > tF_colder
    assert tF_colder < tC_boil
    
    # Greater/Less than or equal
    assert tC_boil <= tF_boil
    assert tC_boil >= tF_boil
    assert tC_boil <= tF_hotter
    assert tF_hotter >= tC_boil


def test_delta_plus_affine_radd(degC, dK):
    """Tests Vector + Point (reflected operation)"""
    t = 10 * degC   # 283.15 K
    d = 5 * dK     # 5 K delta
    
    # Should be identical to t + d
    out = d + t 
    assert isinstance(out, AffineQuantity)
    assert isclose(out._mag_si, 288.15, rel_tol=0, abs_tol=1e-12)
    assert isclose(out.value, 15.0, rel_tol=0, abs_tol=1e-12)
    assert out.unit is degC

def test_delta_minus_affine_raises(degC, dK):
    """Tests that Vector - Point is an invalid operation."""
    t = 10 * degC
    d = 5 * dK
    with pytest.raises(TypeError):
        _ = d - t

# ----------------------------
# Happy paths (should succeed)
# ----------------------------

def test_R_uses_textbook_K_in_denominator_but_linearizes_to_deltaK():
    """
    Textbook expression:
        R = 8.314 * J / (mol * K)
    should now work and internally become J/(mol·ΔK).
    """
    R = 8.314 * u.J / (u.mol * u.K)  # previously raised; now OK
    assert isinstance(R, LinearQuantity)
    # Unit name should include ΔK (linearized); K should not appear as an affine symbol here.
    assert "ΔK" in R.unit.name
    # sanity on dimensions via a quick usage
    T = (101325 * u.Pa) * (0.01 * u.m**3) / (1 * u.mol) / R  # -> ΔK
    assert isinstance(T, LinearQuantity)  # still a delta (difference)


def test_specific_heat_capacity_linearizes_K_in_denominator():
    """
    c_p often written as J/(kg·K) in textbooks.
    Should be linearized to J/(kg·ΔK) and usable in Q = m c ΔT.
    """
    cp = 1000 * u.J / (u.kg * u.K)  # 1000 J/(kg·K)
    assert isinstance(cp, LinearQuantity)
    assert "ΔK" in cp.unit.name

    m = 2 * u.kg
    dT = 15 * u.delta_k
    Q = m * cp * dT
    assert isinstance(Q, LinearQuantity)
    assert "J" in Q.unit.name


def test_linearizes_celsius_in_denominator_too():
    """
    Using °C in denominators should linearize to Δ°C.
    """
    X = 1 * u.J / (u.mol * u.degC)
    assert isinstance(X, LinearQuantity)
    assert "Δ°C" in X.unit.name


def test_alias_kelvin_in_denominator_linearizes():
    """
    Aliases like 'kelvin' should behave like 'K' in composition.
    """
    R2 = 8.314 * u.J / (u.mol * u.kelvin)
    assert isinstance(R2, LinearQuantity)
    assert "ΔK" in R2.unit.name


def test_nested_composition_linearizes_affine_in_denominator():
    """
    More complex mix still linearizes the affine temperature unit in the denominator.
    """
    Y = 10 * (u.N * u.m) / (u.mol * u.K)  # N·m = J
    assert isinstance(Y, type(1 * u.J / u.mol))
    assert "ΔK" in Y.unit.name
    # And dimensions equivalent to J/(mol·ΔK)
    # Using it quickly:
    _ = 5 * Y  # should not raise


# -----------------------------------------
# Guard rails (what should still be blocked)
# -----------------------------------------

def test_plain_unit_reciprocal_of_affine_is_still_blocked():
    """
    Linearization only applies inside the simplifier’s composite-building path.
    Direct unit algebra on an affine unit remains forbidden.
    """
    with pytest.raises(TypeError):
        _ = 1 / u.K  # AffineUnit.__rtruediv__ must still raise


def test_plain_unit_pow_of_affine_is_still_blocked():
    """
    Users should not directly exponentiate an affine unit.
    (Linearization happens only in the composition path.)
    """
    with pytest.raises(TypeError):
        _ = u.K ** 2


def test_quantity_mul_div_with_absolute_temperature_still_blocked():
    """
    Absolute temperatures (AffineQuantity) must remain non-multipliable/dividable.
    """
    T_abs = 300 * u.K
    with pytest.raises(TypeError):
        _ = T_abs * (1 * u.s)
    with pytest.raises(TypeError):
        _ = (1 * u.s) * T_abs
    with pytest.raises(TypeError):
        _ = (1 * u.s) / T_abs


def test_linear_times_affine_quantity_still_blocked():
    """
    Regression guard for LinearQuantity ⨯/÷ AffineQuantity denial.
    """
    T_abs = 300 * u.K
    d = 1 * u.s
    with pytest.raises(TypeError):
        _ = d * T_abs
    with pytest.raises(TypeError):
        _ = d / T_abs


# -------------------------------------------------
# Integration: result remains a delta until anchored
# -------------------------------------------------

def test_resulting_delta_to_absolute_conversion_is_easy():
    """
    Composition uses ΔK to keep algebra linear; turning a delta into an absolute K
    should still be explicit and easy.
    """
    R = 8.314 * u.J / (u.mol * u.K)
    dT = (101325 * u.Pa) * (0.01 * u.m**3) / (1 * u.mol) / R
    assert isinstance(dT, LinearQuantity)  # ΔK

    # Add to an absolute baseline to get AffineQuantity in K
    T_abs = (0 * u.K) + dT
    assert isinstance(T_abs, AffineQuantity)
    assert isclose(T_abs._mag_si, dT._mag_si, abs_tol=1e-12)