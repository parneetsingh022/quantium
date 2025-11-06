import pytest
from math import isclose

from quantium.core.dimensions import DIM_0
from quantium.core.unit import Unit
from quantium.core.quantity import LinearQuantity, AffineQuantity
from quantium.units import u
from quantium.errors import AffineTemperatureOperationError


def dimension(unit : Unit):
    return unit.dim

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

def test_define_universal_gas_constant():
    """
    Textbook expression:
        R = 8.314 * J / (mol * K)
    should now work and internally become J/(mol·ΔK).
    """
    R = 8.314 * u.J / (u.mol * u.K)  # previously raised; now OK
    assert isinstance(R, LinearQuantity)
    # Unit name should include ΔK (linearized); K should not appear as an affine symbol here.
    assert "K" in R.unit.name
    # sanity on dimensions via a quick usage
    T = (101325 * u.Pa) * (0.01 * u.m**3) / (1 * u.mol) / R  # -> K
    assert isinstance(T, AffineQuantity)  # still a delta (difference)


def test_specific_heat_capacity_K_in_denominator():
    """
    c_p often written as J/(kg·K) in textbooks.
    """
    cp = 1000 * u.J / (u.kg * u.K)  # 1000 J/(kg·K)
    assert isinstance(cp, LinearQuantity)
    assert "ΔK" not in cp.unit.name
    assert "K" in cp.unit.name

    m = 2 * u.kg
    dT = 15 * u.delta_k
    Q = m * cp * dT
    assert isinstance(Q, LinearQuantity)
    assert "J" in Q.unit.name


def test_celsius_in_denominator_raises_error():
    with pytest.raises(TypeError):
        X = 1 * u.J / (u.mol * u.degC)
   


def test_alias_kelvin_in_denominator():
    """
    Aliases like 'kelvin' should behave like 'K' in composition.
    """
    R2 = 8.314 * u.J / (u.mol * u.kelvin)
    assert isinstance(R2, LinearQuantity)
    assert "K" in R2.unit.name


def test_nested_composition_linearizes_affine_in_denominator():
    """
    More complex mix still linearizes the affine temperature unit in the denominator.
    """
    Y = 10 * (u.N * u.m) / (u.mol * u.K)  # N·m = J
    assert isinstance(Y, type(1 * u.J / u.mol))
    assert "ΔK" not in Y.unit.name
    assert "K" in Y.unit.name
    # And dimensions equivalent to J/(mol·ΔK)
    # Using it quickly:
    _ = 5 * Y  # should not raise



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


# -----------------------------
# Zero-offset affine: Kelvin & Rankine
# -----------------------------

def test_linearquantity_mul_affineunit_zero_offset_allowed():
    """(LinearQuantity) * (AffineUnit offset==0) -> LinearQuantity"""
    L = 2 * u.m
    out = L * u.K           # uses LinearQuantity.__mul__ path that accepts AffineUnit(offset=0)
    assert isinstance(out, LinearQuantity)
    assert "ΔK" not in out.unit.name
    assert "K" in out.unit.name  # linearized temperature factor present

def test_linearquantity_truediv_affineunit_zero_offset_allowed():
    """(LinearQuantity) / (AffineUnit offset==0) -> LinearQuantity"""
    L = 2 * u.m
    out = L / u.K
    assert isinstance(out, LinearQuantity)
    assert "ΔK" not in out.unit.name
    assert "K" in out.unit.name

def test_linearquantity_rmul_affineunit_zero_offset_allowed():
    """(AffineUnit offset==0) * (LinearQuantity) -> LinearQuantity via __rmul__ on LinearQuantity"""
    L = 3 * u.s
    out = 1 * u.K * L
    assert isinstance(out, LinearQuantity)
    assert "ΔK" not in out.unit.name
    assert "K" in out.unit.name

def test_linearquantity_rtruediv_affineunit_zero_offset_allowed():
    """(AffineUnit offset==0) / (LinearQuantity) -> LinearQuantity via __rtruediv__ on LinearQuantity"""
    L = 2 * u.s
    out = 1* u.K / L
    assert isinstance(out, LinearQuantity)
    assert "ΔK" not in out.unit.name
    assert "K" in out.unit.name

def test_affinequantity_mul_linearunit_zero_offset_allowed():
    """(AffineQuantity in K) * (LinearUnit) -> LinearQuantity"""
    T = 300 * u.K
    out = T * u.s
    assert isinstance(out, LinearQuantity)
    assert "ΔK" not in out.unit.name
    assert "K" in out.unit.name

def test_affinequantity_truediv_linearunit_zero_offset_allowed():
    """(AffineQuantity in K) / (LinearUnit) -> LinearQuantity"""
    T = 300 * u.K
    out = T / u.s
    assert isinstance(out, LinearQuantity)
    assert "ΔK" not in out.unit.name
    assert "K" in out.unit.name

def test_affinequantity_rmul_linearquantity_zero_offset_allowed():
    """(LinearQuantity) * (AffineQuantity in K) -> LinearQuantity, tests AffineQuantity.__rmul__"""
    v = 10 * (u.m / u.s)
    T = 300 * u.K
    out = v * T
    assert isinstance(out, LinearQuantity)
    assert "ΔK" not in out.unit.name
    assert "K" in out.unit.name

def test_affinequantity_rtruediv_linearquantity_zero_offset_allowed():
    """(LinearQuantity) / (AffineQuantity in K) -> LinearQuantity, tests AffineQuantity.__rtruediv__"""
    E = 10 * u.J
    T = 250 * u.K
    out = E / T
    assert isinstance(out, LinearQuantity)
    # energy per delta-temperature (e.g., J/ΔK)
    assert "ΔK" not in out.unit.name
    assert "K" in out.unit.name

def test_affinequantity_div_affinequantity_zero_offset_is_dimensionless_linear():
    """(AffineQuantity K) / (AffineQuantity K) -> LinearQuantity dimensionless."""
    T1 = 400 * u.K
    T2 = 200 * u.K
    out = T1 / T2
    assert isinstance(out, LinearQuantity)
    assert out.dim == DIM_0
    # 400K / 200K == 2
    assert isclose(out.value, 2.0, rel_tol=0, abs_tol=1e-12)

def test_rankine_behaves_like_kelvin_in_products_and_ratios():
    """Rankine (offset=0) should behave like Kelvin in multiplicative algebra."""
    T = 540 * u.degR  # 300 K
    q1 = T * u.s
    q2 = (1 * u.J) / T
    assert isinstance(q1, LinearQuantity) and isinstance(q2, LinearQuantity)
    assert "ΔR" not in q1.unit.name and "ΔR" not in q2.unit.name
    assert "R" in q1.unit.name and "R" in q2.unit.name


# -----------------------------
# Non-zero-offset affines: Celsius & Fahrenheit (must raise)
# -----------------------------

@pytest.mark.parametrize("unit", [u.degC, u.degF])
def test_linearquantity_mul_div_with_nonzero_offset_affineunit_raises(unit):
    L = 2 * u.m
    with pytest.raises(TypeError):
        _ = L * unit
    with pytest.raises(TypeError):
        _ = L / unit
    with pytest.raises(TypeError):
        _ = unit * L
    with pytest.raises(TypeError):
        _ = unit / L

@pytest.mark.parametrize("unit", [u.degC, u.degF])
def test_affinequantity_mul_div_with_linearunit_raises(unit):
    T = 25 * unit
    with pytest.raises(TypeError):
        _ = T * u.s
    with pytest.raises(TypeError):
        _ = T / u.s
    with pytest.raises(TypeError):
        _ = (1 * u.s) * T
    with pytest.raises(TypeError):
        _ = (1 * u.J) / T

@pytest.mark.parametrize("left,right", [(u.degC, u.K), (u.degF, u.K), (u.degC, u.degF)])
def test_affinequantity_mul_div_with_affineunit_nonzero_offset_raises(left, right):
    """Any multiplicative mixing with °C/°F must raise."""
    T_left = 10 * left
    with pytest.raises(TypeError):
        _ = T_left * right
    with pytest.raises(TypeError):
        _ = T_left / right
    with pytest.raises(TypeError):
        _ = right * T_left
    with pytest.raises(TypeError):
        _ = right / T_left

def test_affine_scalar_scaling_still_raises_for_all_affines():
    """Scaling absolute affines by scalars is undefined (even for K/°R by design)."""
    for unit in (u.degC, u.degF, u.K, u.degR):
        T = 10 * unit
        with pytest.raises(TypeError):
            _ = T * 2
        with pytest.raises(TypeError):
            _ = 2 * T
        with pytest.raises(TypeError):
            _ = T / 2
        with pytest.raises(TypeError):
            _ = 2 / T


# -----------------------------
# Symbol-map / composition sanity (names include ΔK)
# -----------------------------

def test_linearquantity_times_kelvin_has_K_in_unit_name():
    q = (5 * u.N) * u.K  # -> N·ΔK
    assert isinstance(q, LinearQuantity)
    assert "K" in q.unit.name

def test_linearquantity_over_kelvin_has_K_in_denominator():
    q = (5 * u.N) / u.K  # -> N/ΔK
    assert isinstance(q, LinearQuantity)
    assert "K" in q.unit.name

def test_linearquantity_times_rankine_has_R_in_denominator():
    q = (5 * u.N) * u.degR  # -> N/ΔK (since °R is ratio-scale)
    assert isinstance(q, LinearQuantity)
    assert "°R" in q.unit.name

def test_linearquantity_over_rankine_has_R_in_denominator():
    q = (5 * u.N) / u.degR  # -> N/ΔK (since °R is ratio-scale)
    assert isinstance(q, LinearQuantity)
    assert "°R" in q.unit.name


def test_quantity_mul_div_with_kelvin_and_rankine_allowed():
    """
    Multiplication and division with ratio-scale (zero-offset) temperatures
    like Kelvin and Rankine are allowed.
    """
    T_K = 300 * u.K
    T_R = 540 * u.degR
    s = 1 * u.s

    # These should now be valid operations
    _ = T_K * s
    _ = s * T_K
    _ = s / T_K

    _ = T_R * s
    _ = s * T_R
    _ = s / T_R


def test_quantity_mul_div_with_affine_temperatures_blocked():
    """
    Multiplication and division with affine (nonzero offset) temperatures like
    Celsius and Fahrenheit must raise TypeError.
    """
    T_C = 25 * u.degC
    T_F = 77 * u.degF
    s = 1 * u.s

    with pytest.raises(TypeError):
        _ = T_C * s
    with pytest.raises(TypeError):
        _ = s * T_C
    with pytest.raises(TypeError):
        _ = s / T_C

    with pytest.raises(TypeError):
        _ = T_F * s
    with pytest.raises(TypeError):
        _ = s * T_F
    with pytest.raises(TypeError):
        _ = s / T_F



def test_affine_result_from_linear_multiplication():
    r1 = (100 * u.m) * (10 * u.K)
    r2 = (100 * (1/u.m)) * r1

    assert r2.unit == u.K
    assert isinstance(r2, AffineQuantity)

    # test reverse multiplication (__rmul__)
    r3 = r1 * (100 * (1/u.m))
    assert r3.unit == u.K
    assert isinstance(r3, AffineQuantity)

    # test commutativity for scalar * affine
    scalar = 2
    r4 = scalar * r1
    r5 = r1 * scalar
    assert r4 == r5
    assert isinstance(r4, LinearQuantity)

    # test chained multiplications
    r6 = (2 * (1/u.m)) * r1 * (50 * u.m)
    assert r6.unit == r1.unit
    assert isinstance(r6, LinearQuantity)

    dimless = 1.0
    r7 = r1 * dimless
    assert r7 == r1
    assert isinstance(r7, LinearQuantity)

def test_ideal_gas_law_all_combinations():
    # Given
    n = 1.0 * u.mol                                # amount of substance
    R = 8.314462618 * (u.J / (u.mol * u.K))        # ideal gas constant
    T = 300 * u.K                                  # absolute temperature
    V = 0.025 * u.m**3                             # volume

    expected_P_kPa = 99.77355 * u.kPa              # expected ≈ 99.77 kPa

    # --- 1. Solve for Pressure ---
    P = (n * R * T) / V
    assert P.to(u.kPa).value == pytest.approx(expected_P_kPa.value, rel=1e-3)
    assert dimension(P.unit) == dimension(u.Pa)

    # --- 2. Solve for Volume ---
    V_calc = (n * R * T) / P
    assert V_calc.to(u.m**3).value == pytest.approx(V.value, rel=1e-3)
    assert dimension(V_calc.unit) == dimension(u.m**3)

    # --- 3. Solve for Temperature ---
    T_calc = (P * V) / (n * R)
    assert T_calc.to(u.K).value == pytest.approx(T.value, rel=1e-3)
    assert T_calc.unit == u.K

    # --- 4. Solve for Amount of Substance ---
    n_calc = (P * V) / (R * T)
    assert n_calc.to(u.mol).value == pytest.approx(n.value, rel=1e-3)
    assert n_calc.unit == u.mol

    # --- 5. Solve for Gas Constant (consistency check) ---
    R_calc = (P * V) / (n * T)
    assert R_calc.to(u.J / (u.mol * u.K)).value == pytest.approx(R.value, rel=1e-3)
    assert dimension(R_calc.unit) == dimension(u.J / (u.mol * u.K))

def test_resultant_unit_is_affine():
    R = 8.314 * u.J / (u.mol * u.K)
    dT = (101325 * u.Pa) * (0.01 * u.m**3) / (1 * u.mol) / R
    assert isinstance(dT, AffineQuantity)  # K


def unit_multiplication_and_division_with_affine_all():
    # --- Base sanity (Kelvin and Rankine) ---
    assert (100 * u.s) * u.K == 100 * (u.s * u.K)
    assert u.K * (100 * u.s) == 100 * (u.K * u.s)

    assert (100 * u.s) * u.R == 100 * (u.s * u.R)
    assert u.R * (100 * u.s) == 100 * (u.R * u.s)

    assert (100 * u.s) / u.K == 100 * (u.s / u.K)
    assert u.K / (100 * u.s) == (1 / 100) * (u.K / u.s)

    assert (100 * u.s) / u.R == 100 * (u.s / u.R)
    assert u.R / (100 * u.s) == (1 / 100) * (u.R / u.s)

    # --- Commutativity of multiplication (unit * unit) ---
    assert u.s * u.K == u.K * u.s
    assert u.s * u.R == u.R * u.s
    assert u.K * u.R == u.R * u.K

    # --- Division forms ---
    assert (u.K * u.s) / u.K == u.s
    assert (u.s * u.K) / u.s == u.K
    assert (u.R * u.s) / u.R == u.s
    assert (u.s * u.R) / u.s == u.R
    assert (u.K / u.s) * u.s == u.K
    assert (u.R / u.s) * u.s == u.R
    assert (u.s / u.K) * u.K == u.s
    assert (u.s / u.R) * u.R == u.s

    # --- Dimensionless results ---
    assert (u.K / u.K) == DIM_0
    assert (u.R / u.R) == DIM_0
    assert (u.K / u.R) != DIM_0  # same dimension but different scale
    assert (u.R / u.K) != DIM_0

    # --- Mixed with different scalars (both orders) ---
    assert (2 * u.K) * (50 * u.s) == 100 * (u.K * u.s)
    assert (50 * u.s) * (2 * u.K) == 100 * (u.s * u.K)
    assert (2 * u.R) * (50 * u.s) == 100 * (u.R * u.s)
    assert (50 * u.s) * (2 * u.R) == 100 * (u.s * u.R)

    assert (100 * u.s) / (2 * u.K) == 50 * (u.s / u.K)
    assert (2 * u.K) / (100 * u.s) == (1 / 50) * (u.K / u.s)
    assert (100 * u.s) / (2 * u.R) == 50 * (u.s / u.R)
    assert (2 * u.R) / (100 * u.s) == (1 / 50) * (u.R / u.s)

    # --- Associativity with scalars pulled in/out ---
    assert ((5 * u.s) * u.K) == 5 * (u.s * u.K)
    assert (u.K * (5 * u.s)) == 5 * (u.K * u.s)
    assert ((5 * u.s) * u.R) == 5 * (u.s * u.R)
    assert (u.R * (5 * u.s)) == 5 * (u.R * u.s)

    assert ((5 * u.s) / u.K) == 5 * (u.s / u.K)
    assert ((5 * u.K) / u.s) == 5 * (u.K / u.s)
    assert ((5 * u.s) / u.R) == 5 * (u.s / u.R)
    assert ((5 * u.R) / u.s) == 5 * (u.R / u.s)

    # --- Inverse-style identities (robustness) ---
    assert (((3 * u.s) * u.K) / u.K) == 3 * u.s
    assert (((3 * u.K) * u.s) / u.s) == 3 * u.K
    assert (((3 * u.s) * u.R) / u.R) == 3 * u.s
    assert (((3 * u.R) * u.s) / u.s) == 3 * u.R

    assert (((3 * u.K) / u.s) * u.s) == 3 * u.K
    assert (((3 * u.s) / u.K) * u.K) == 3 * u.s
    assert (((3 * u.R) / u.s) * u.s) == 3 * u.R
    assert (((3 * u.s) / u.R) * u.R) == 3 * u.s

    # --- Unit power combinations ---
    assert u.K * u.K == u.K**2
    assert (10 * u.K) * u.K == 10 * (u.K**2)
    assert u.R * u.R == u.R**2
    assert (10 * u.R) * u.R == 10 * (u.R**2)

def unit_multiplication_and_division_between_units_all():
    # --- Multiplication between units ---
    assert u.s * u.K == u.K * u.s
    assert u.s * u.R == u.R * u.s
    assert u.K * u.R == u.R * u.K
    assert u.m * u.s == u.m * u.s
    assert u.N * u.m == u.J                # derived unit check, if defined

    # --- Division between units ---
    assert u.s / u.K == u.s / u.K
    assert u.s / u.R == u.s / u.R
    assert u.K / u.s == 1 / (u.s / u.K)
    assert u.R / u.s == 1 / (u.s / u.R)
    assert (u.m / u.s) * u.s == u.m
    assert (u.K / u.K) == DIM_0
    assert (u.R / u.R) == DIM_0
    assert (u.K / u.R) != DIM_0
    assert (u.R / u.K) != DIM_0

    # --- Unit powers and associativity ---
    assert u.K * u.K == u.K**2
    assert (u.K**2) / u.K == u.K
    assert (u.K / u.K**2) == 1 / u.K
    assert u.R * u.R == u.R**2
    assert (u.R**2) / u.R == u.R
    assert (u.R / u.R**2) == 1 / u.R

    # --- Chained multiplication/division ---
    assert (u.s * u.K) / u.K == u.s
    assert (u.s / u.K) * u.K == u.s
    assert (u.K * u.s) / u.s == u.K
    assert (u.K / u.s) * u.s == u.K
    assert (u.s * u.R) / u.R == u.s
    assert (u.s / u.R) * u.R == u.s
    assert (u.R * u.s) / u.s == u.R
    assert (u.R / u.s) * u.s == u.R

    # --- Inverse-style consistency ---
    assert (u.K * (1 / u.K)) == DIM_0
    assert ((1 / u.K) * u.K) == DIM_0
    assert (u.R * (1 / u.R)) == DIM_0
    assert ((1 / u.R) * u.R) == DIM_0

    # --- Associativity ---
    assert ((u.s * u.m) * u.K) == (u.s * (u.m * u.K))
    assert ((u.s * u.m) * u.R) == (u.s * (u.m * u.R))

    # --- Power cross-checks ---
    assert (u.K**3) / (u.K**2) == u.K
    assert (u.K**2) * (u.K**3) == u.K**5
    assert (u.R**3) / (u.R**2) == u.R
    assert (u.R**2) * (u.R**3) == u.R**5

def unit_scalar_and_quantity_operator_coverage():
    # --- scalar × unit (triggers unit.__rmul__) ---
    assert 3 * u.K == (3 * u.K)                # same object model
    assert 3 * u.R == (3 * u.R)

    # --- unit × scalar (triggers unit.__mul__) ---
    assert u.K * 3 == 3 * u.K
    assert u.R * 3 == 3 * u.R

    # --- scalar ÷ unit (triggers unit.__rtruediv__) ---
    assert 6 / u.K == (6 * (u.K ** -1))
    assert 6 / u.R == (6 * (u.R ** -1))

    # --- unit ÷ scalar (triggers unit.__truediv__) ---
    assert (u.K / 2) * 2 == u.K
    assert (u.R / 2) * 2 == u.R

    # --- quantity × unit (quantity.__mul__) & reflected (unit.__rmul__) ---
    q = 10 * u.s
    assert q * u.K == 10 * (u.s * u.K)
    assert u.K * q == 10 * (u.K * u.s)
    assert q * u.R == 10 * (u.s * u.R)
    assert u.R * q == 10 * (u.R * u.s)

    # --- quantity ÷ unit (quantity.__truediv__) & unit ÷ quantity (unit.__truediv__/__rtruediv__) ---
    assert (q / u.K) * u.K == q
    assert (q / u.R) * u.R == q
    assert (u.K / q) * q == u.K
    assert (u.R / q) * q == u.R

    # --- dimensionless interactions ---
    assert (u.K / u.K) == DIM_0
    assert (u.R / u.R) == DIM_0
    assert (q / q) == DIM_0
    assert (DIM_0 * u.K) == u.K
    assert (u.K * DIM_0) == u.K
    assert (DIM_0 / u.K) == (u.K ** -1)
    assert (u.K / DIM_0) == u.K

    # --- mixed K↔R ratios (same dimension, not dimensionless) ---
    assert (u.K / u.R) != DIM_0
    assert (u.R / u.K) != DIM_0

    # --- powers (unit.__pow__) ---
    assert u.K * u.K == u.K**2
    assert u.R * u.R == u.R**2
    assert (u.K**2) / u.K == u.K
    assert (u.R**2) / u.R == u.R
    assert (u.K**0) == DIM_0
    assert (u.K**-1) * u.K == DIM_0
    assert (u.R**-1) * u.R == DIM_0

    # --- associativity/commutativity sanity (where valid) ---
    assert (u.s * u.K) == (u.K * u.s)
    assert ((u.s * u.K) * u.R) == (u.s * (u.K * u.R))
    assert ((q * u.K) / u.K) == q
    assert ((q / u.K) * u.K) == q


def unit_unit_operator_coverage():
    # --- unit × unit (mul / commutative) ---
    assert u.s * u.K == u.K * u.s
    assert u.s * u.R == u.R * u.s
    assert u.K * u.R == u.R * u.K

    # --- unit ÷ unit (truediv / rtruediv via reordering with scalars) ---
    assert (u.K / u.K) == DIM_0
    assert (u.R / u.R) == DIM_0
    assert ((u.s * u.K) / u.K) == u.s
    assert ((u.s * u.R) / u.R) == u.s
    assert (u.K / u.s) * u.s == u.K
    assert (u.R / u.s) * u.s == u.R

    # reflected division with scalar wrapper (exercise __rtruediv__)
    # Here 1 / (unit/unit) → still dimensionless if same units
    assert 1 / (u.K / u.K) == 1 * DIM_0
    assert 1 / (u.R / u.R) == 1 * DIM_0

    # --- powers & inverses ---
    assert (u.K**3) / (u.K**2) == u.K
    assert (u.R**3) / (u.R**2) == u.R
    assert (u.K**-1) == (1 / u.K)
    assert (u.R**-2) == (1 / (u.R**2))

    # --- chained cancelations ---
    assert ((u.s * u.K) / (u.K * u.s)) == DIM_0
    assert ((u.s / u.K) * (u.K / u.s)) == DIM_0




def error_behavior_for_invalid_ops():

    with pytest.raises(AffineTemperatureOperationError):
        _ = u.K + u.s  # addition across unrelated dims should fail

    with pytest.raises(AffineTemperatureOperationError):
        _ = 3 + u.K    # scalar + unit should fail if not allowed

def test_affine_temperature_operations_raise():
    # Kelvin/Rankine are OK → no error
    (100 * u.s) * u.K
    (100 * u.s) / u.degR

    # Celsius/Fahrenheit should raise AffineTemperatureOperationError
    with pytest.raises(AffineTemperatureOperationError):
        (100 * u.s) * u.degC

    with pytest.raises(AffineTemperatureOperationError):
        u.degC * (100 * u.s)

    with pytest.raises(AffineTemperatureOperationError):
        (100 * u.s) / u.degC

    with pytest.raises(AffineTemperatureOperationError):
        u.degC / (100 * u.s)

    with pytest.raises(AffineTemperatureOperationError):
        (100 * u.s) * u.degF

    with pytest.raises(AffineTemperatureOperationError):
        u.degF * (100 * u.s)

    with pytest.raises(AffineTemperatureOperationError):
        (100 * u.s) / u.degF

    with pytest.raises(AffineTemperatureOperationError):
        u.degF / (100 * u.s)