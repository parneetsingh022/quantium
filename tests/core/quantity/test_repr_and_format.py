from math import isclose
import pytest

from quantium.core.dimensions import LENGTH, MASS, TIME
from quantium.core.unit import LinearUnit
from quantium.units.registry import DEFAULT_REGISTRY as ureg
import quantium.core.utils as utils
from quantium.units import u



from tests.utils import _nop_prettifier
from quantium.core import utils


TIME_INV = TIME ** -1

# -------------------------------
# __repr__: pretty printing
# -------------------------------

def test_repr_keeps_non_si_unit_name(monkeypatch):
    # Ensure repr uses the current unit name ("cm") and does not replace with a symbol
    import quantium.core.utils as utils

    # Make prettifier a no-op passthrough so we can assert on exact string.
    monkeypatch.setattr(utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True)
    # Ensure it would *try* to upgrade only when scale_to_si == 1.0; here it's 0.01, so no upgrade.
    monkeypatch.setattr(utils, "preferred_symbol_for_dim", lambda d: "m", raising=True)

    cm = LinearUnit("cm", 0.01, LENGTH)
    q = 2 * cm
    assert repr(q) == "2 cm"

def test_repr_upgrades_to_preferred_symbol_when_scale_is_1(monkeypatch):
    from quantium import core as _core
    # prettifier just returns what it's given
    monkeypatch.setattr(
        _core.utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True
    )
    # preferred symbol for LENGTH is "m"
    monkeypatch.setattr(_core.utils, "preferred_symbol_for_dim", lambda d: "m" if d == LENGTH else None, raising=True)

    m = LinearUnit("m", 1.0, LENGTH)
    q = 3 * m
    # scale_to_si == 1.0 -> allowed to upgrade pretty name to "m"
    assert repr(q) == "3 m"


def test_repr_consistent_for_equivalent_component_units(monkeypatch):
    _nop_prettifier(monkeypatch)

    q1 = 100 * (u.V / u.m)
    q2 = 100 * (u.W / (u.A * u.m))
    q3 = 100 * u("V/m")
    q4 = 100 * u("W/(A*m)")

    assert repr(q1) == "100 V/m"
    assert repr(q3) == "100 V/m"

    assert repr(q2) == "100 W/(A·m)"
    assert repr(q4) == "100 W/(A·m)"


# -------------------------------
# __format__: formatting specs (broad coverage)
# -------------------------------


def test_format_equivalents_across_units(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    cm = ureg.get("cm")
    s  = ureg.get("s")
    kPa = ureg.get("kPa")
    ohm = ureg.get("ohm")  # alias -> Ω
    min_ = ureg.get("min")

    # 1) Velocity in non-SI (cm/s)
    v = 250 * (cm / s)
    assert f"{v}" == "250 cm/s"
    assert f"{v:native}" == "250 cm/s"

    # 2) Pressure with prefix (kPa)
    p = 2 * kPa
    assert f"{p}" == "2 kPa"
    assert f"{p:native}" == "2 kPa"

    # 3) Alias normalization (ohm → Ω)
    r = 5 * ohm
    assert f"{r}" == "5 Ω"
    assert f"{r:native}" == "5 Ω"

    # 4) Mixed time unit in denominator (m/min)
    speed = 120 * (ureg.get("m") / min_)
    assert f"{speed}" == "120 m/min"
    assert f"{speed:native}" == "120 m/min"


def test_format_si_converts_various_units(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    cm = ureg.get("cm")
    s  = ureg.get("s")
    kPa = ureg.get("kPa")
    min_ = ureg.get("min")

    # 1) 1000 cm/s -> 10 m/s
    v = 1000 * (cm / s)
    assert f"{v:si}" == "10 m/s"

    # 2) 2 kPa -> 2000 Pa
    p = 2 * kPa
    assert f"{p:si}" == "2000 Pa"

    # 3) 120 m/min -> 2 m/s (since 1 min = 60 s)
    speed = 120 * (ureg.get("m") / min_)
    assert f"{speed:si}" == "2 m/s"

    # 4) Frequency with prefix: 2 kHz -> 2000 Hz
    kHz = ureg.get("kHz")
    f = 2 * kHz
    assert f"{f:si}" == "2000 Hz"


def test_format_with_micro_and_normalization(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    # Leading 'u' maps to Greek micro 'µ' during lookup/registration
    uF = ureg.get("uF")     # normalized to µF internally
    q = 3 * uF
    # Current unit uses the canonical symbol
    assert f"{q}" == "3 µF"
    # SI format converts to base Farads (3e-06 F)
    assert f"{q:si}" == "3e-06 F"


def test_format_whitespace_and_case_insensitivity(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    cm = ureg.get("cm")
    s  = ureg.get("s")
    v = 1000 * (cm / s)  # 10 m/s in SI

    # Varied spacing/casing should still resolve to SI
    assert f"{v: SI }" == "10 m/s"
    assert f"{v:Si}" == "10 m/s"
    assert f"{v:  sI}" == "10 m/s"


def test_format_dimensionless_is_numeric_only(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    # Dimensionless via equal ratio (kPa / Pa)
    kPa = ureg.get("kPa")
    Pa  = ureg.get("Pa")

    q = (3 * kPa) / (3000 * Pa)  # equals 1 (dimensionless)
    assert f"{q}" == "1"
    assert f"{q:native}" == "1"
    assert f"{q:si}" == "1"


def test_format_invalid_spec_raises(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    m = ureg.get("m")
    q = 3 * m
    with pytest.raises(ValueError):
        _ = f"{q:unknown}"

# -------------------------------
# .si preserves correct SI family (Hz/Bq, Gy/Sv)
# -------------------------------

def test_si_preserves_family_for_time_inverse_and_dose(monkeypatch):
    _nop_prettifier(monkeypatch)
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()

    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    Hz = ureg.get("Hz")
    Bq = ureg.get("Bq")
    Gy = ureg.get("Gy")
    Sv = ureg.get("Sv")

    kHz = ureg.get("kHz")
    kBq = ureg.get("kBq")
    kGy = ureg.get("kGy")
    kSv = ureg.get("kSv")

    # Sanity: original units print as-is
    assert f"{100 * Hz}" == "100 Hz"
    assert f"{100 * Bq}" == "100 Bq"
    assert f"{100 * Gy}" == "100 Gy"
    assert f"{100 * Sv}" == "100 Sv"

    # Prefixed forms print as-is
    assert f"{100 * kHz}" == "100 kHz"
    assert f"{100 * kBq}" == "100 kBq"
    assert f"{100 * kGy}" == "100 kGy"
    assert f"{100 * kSv}" == "100 kSv"

    # .si should preserve family heads (Hz↔Hz, Bq↔Bq, Gy↔Gy, Sv↔Sv)
    assert f"{(100 * kHz):si}" == "100000 Hz"
    assert f"{(100 * kBq):si}" == "100000 Bq"
    assert f"{(100 * kGy):si}" == "100000 Gy"
    assert f"{(100 * kSv):si}" == "100000 Sv"


# -------------------------------
# __repr__ should not flip atomic symbols but upgrade composed SI names
# -------------------------------

def test_repr_preserves_atomic_symbols_and_prefixed(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    # Atomic SI units stay unchanged
    for sym in ("Hz", "Bq", "Gy", "Sv"):
        u = ureg.get(sym)
        assert f"{100 * u}" == f"100 {sym}"

    # Prefixed atomic SI units also stay unchanged
    for sym in ("kHz", "kBq", "kGy", "kSv"):
        u = ureg.get(sym)
        assert f"{100 * u}" == f"100 {sym}"


def test_repr_upgrades_only_composed_si(monkeypatch):
    _nop_prettifier(monkeypatch)
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()

    from quantium.units.registry import DEFAULT_REGISTRY as ureg
    C = ureg.get("C")
    s = ureg.get("s")
    kg = ureg.get("kg")
    m = ureg.get("m")

    # C/s -> A (preferred symbol for electric current)
    q_current = (1 * C) / (1 * s)
    assert f"{q_current}" == "1 A"

    # kg·m/s² -> N (preferred symbol for force)
    # wrap s as a LinearQuantity: (1 * s) ** 2, not (s ** 2)
    q_force = (2 * kg) * (3 * m) / ((1 * s) ** 2)
    assert f"{q_force}" == "6 N"



def test_si_fallback_to_composed_when_no_named_symbol(monkeypatch):
    _nop_prettifier(monkeypatch)
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()

    from quantium.units.registry import DEFAULT_REGISTRY as ureg
    cm = ureg.get("cm")
    s = ureg.get("s")

    v = 1000 * (cm / s)  # velocity: no named SI symbol
    assert f"{v:si}" == "10 m/s"


def test_force_micro_and_kilo_newton(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    mg = ureg.get("mg")    # 1e-6 kg
    um = ureg.get("µm")    # 1e-6 m
    ms = ureg.get("ms")    # 1e-3 s
    # mg·µm/ms² -> (1e-6*1e-6)/(1e-6) = 1e-6 -> µN
    q_micro = 1 * (mg * um / (ms ** 2))
    assert f"{q_micro}" == "1 mg·µm/ms^2"

    kg = ureg.get("kg")    # 1 kg
    km = ureg.get("km")    # 1e3 m
    s  = ureg.get("s")     # 1 s
    # kg·km/s² 
    q_kilo = 1 * (kg * km / (s ** 2))
    assert f"{q_kilo}" == "1 kg·km/s^2"


def test_current_symbol_and_prefix(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    C  = ureg.get("C")
    s  = ureg.get("s")
    mC = ureg.get("mC")   # 1e-3 C
    ms = ureg.get("ms")   # 1e-3 s

    # C/s -> A
    q_A = (1 * C) / (1 * s)
    assert f"{q_A}" == "1 A"

    # µC/ms -> (1e-6 / 1e-3) = 1e-3 -> mA
    uC = ureg.get("µC")
    q_mA = (1 * uC) / (1 * ms)
    assert f"{q_mA}" == "1 mA"

    # mC/s -> (1e-3 / 1) = 1e-3 -> mA
    q_mA2 = (1 * mC) / (1 * s)
    assert f"{q_mA2}" == "1 mA"


def test_power_symbol_and_prefix(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    J  = ureg.get("J")
    s  = ureg.get("s")
    mJ = ureg.get("mJ")
    ms = ureg.get("ms")
    uJ = ureg.get("µJ")

    # J/s -> W
    q_W = (1 * J) / (1 * s)
    assert f"{q_W}" == "1 W"

    # mJ/ms -> (1e-3 / 1e-3) = 1 -> W
    q_W2 = (1 * mJ) / (1 * ms)
    assert f"{q_W2}" == "1 W"

    # µJ/ms -> (1e-6 / 1e-3) = 1e-3 -> mW
    q_mW = (1 * uJ) / (1 * ms)
    assert f"{q_mW}" == "1 mW"


def test_pressure_symbol_and_prefix(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    mN = ureg.get("mN")   # 1e-3 N composed from kg·m/s² but available via registry
    mm = ureg.get("mm")   # 1e-3 m

    # mN/mm² -> (1e-3) / (1e-3)^2 = 1e3 -> kPa
    # construct as a LinearQuantity / LinearQuantity so the unit algebra flows through
    q_kPa = (1 * mN) / ((1 * mm) ** 2)
    assert f"{q_kPa}" == "1 kPa"

    # N/mm² -> (1) / (1e-3)^2 = 1e6 -> MPa
    N = ureg.get("N")
    q_MPa = (1 * N) / ((1 * mm) ** 2)
    assert f"{q_MPa}" == "1 MPa"


def test_reciprocal_time_units_and_manual_frequency_conversion(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    ms = ureg.get("ms")
    us = ureg.get("µs")
    s  = ureg.get("s")
    Hz = ureg.get("Hz")
    kHz = ureg.get("kHz")
    MHz = ureg.get("MHz")

    # Base reciprocal units (no auto pretty-print)
    q_inv_s = 1 * (1 / s)
    q_inv_ms = 1 * (1 / ms)
    assert f"{q_inv_s}" == "1 1/s"
    assert f"{q_inv_ms}" == "1 1/ms"

    # Manual conversions to frequency units
    q_Hz = q_inv_s.to(Hz)
    assert f"{q_Hz}" == "1 Hz"

    q_kHz = q_inv_ms.to(kHz)
    assert f"{q_kHz}" == "1 kHz"

    q_MHz = (1 * (1 / us)).to(MHz)
    assert f"{q_MHz}" == "1 MHz"


def test_hz_and_bq_unit_format_and_interconversion(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    Hz = ureg.get("Hz")
    Bq = ureg.get("Bq")

    # Creating quantities directly
    q_Hz = 1 * Hz
    q_Bq = 1 * Bq

    # They should keep their original symbols when printed
    assert f"{q_Hz}" == "1 Hz"
    assert f"{q_Bq}" == "1 Bq"

    # Both have the same physical dimension (1/s), so conversion is possible
    q_Bq_to_Hz = q_Bq.to(Hz)
    q_Hz_to_Bq = q_Hz.to(Bq)

    # After conversion, value should remain 1, and symbol should match target
    assert f"{q_Bq_to_Hz}" == "1 Hz"
    assert f"{q_Hz_to_Bq}" == "1 Bq"


def test_atomic_units_not_flipped(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    # Atomic SI heads should print as-is (no cross-family flip)
    for sym in ("Hz", "Bq", "Gy", "Sv", "Pa", "A", "W", "N"):
        u = ureg.get(sym)
        assert f"{100 * u}" == f"100 {sym}"


def test_dimensionless_prints_number(monkeypatch):
    _nop_prettifier(monkeypatch)
    from quantium.units.registry import DEFAULT_REGISTRY as ureg

    m = ureg.get("m")
    # (1 m) / (1 m) is dimensionless → bare number
    q = (1 * m) / (1 * m)
    # repr should be just the number; tolerate "1" exactly
    assert f"{q}" == "1"


# -------------------------------
# __repr__: Regression tests for cancellation + upgrade logic (Issue #72)
# (These tests use the *real* prettifier)
# -------------------------------

@pytest.mark.regression(reason="Issue #72: LinearQuantity.__repr__ incorrectly upgrades a cancelled unit to a prefixed SI unit (e.g., mg becomes µkg)")
def test_issue72_repr_bug_fix_kg_mg_per_kg(monkeypatch):
    """
    REGRESSION TEST for the "µkg" bug.
    Ensures that a unit like "kg·mg/kg" simplifies to "mg" *first*,
    and then __repr__ does NOT upgrade "mg" to "µkg".
    """
    # Invalidate cache to be safe
    utils.invalidate_preferred_cache()
    
    kg = ureg.get("kg")
    mg = ureg.get("mg")
    
    # This is the exact calculation from the bug report
    dose_rate = 15 * (mg / kg)
    patient_mass = 75 * kg
    
    required_dose = patient_mass * dose_rate
    
    # 1. Check the unit name and scale *after* multiplication
    assert required_dose.unit.name == "kg" 
    
    assert isclose(required_dose.unit.scale_to_si, 1)
    assert required_dose.dim == MASS
    
    # 2. Check the reverse order
    required_dose_rev = dose_rate * patient_mass
    assert required_dose_rev.unit.name == "kg"
    assert isclose(required_dose_rev.unit.scale_to_si, 1)

@pytest.mark.regression(reason="Issue #72: LinearQuantity.__repr__ incorrectly upgrades a cancelled unit to a prefixed SI unit (e.g., mg becomes µkg)")
def test_issue72_repr_cancellation_to_prefixed_si(monkeypatch):
    """
    Tests that cancellation still allows a *correct* upgrade when
    the *simplified* unit is still composed.
    'm·mg/kg' -> 'm·(1e-6 kg) / kg' -> 1e-6 m -> 'µm'
    """
    utils.invalidate_preferred_cache()

    m = ureg.get("m")
    mg = ureg.get("mg")
    kg = ureg.get("kg")
    
    q = (1 * m) * (1 * mg) / (1 * kg) # 1 * (m·mg/kg)
    
    assert f"{q}" == "1 µm"
    assert isclose(q.unit.scale_to_si, 1e-6) # m * (mg/kg) = 1 * 1e-6
    assert q.dim == LENGTH # Dimension is Length
    
    

@pytest.mark.regression(reason="Issue #72: LinearQuantity.__repr__ incorrectly upgrades a cancelled unit to a prefixed SI unit (e.g., mg becomes µkg)")
def test_issue72_repr_cancellation_to_dimensionless(monkeypatch):
    """
    Tests that 'mg/kg' simplifies to a dimensionless number.
    The prettifier returns "1", and __repr__ should only print the number.
    """
    utils.invalidate_preferred_cache()
    
    mg = ureg.get("mg")
    kg = ureg.get("kg")
    
    q = (1 * mg) / (1 * kg) # 1 * (mg/kg)
    
    # Value is 1e-6 kg / 1 kg = 1e-6
    assert isclose(q.value, 1e-6)
    # __repr__ should just be the number
    assert f"{q}" == "1e-06"
    
    # Test with equal values
    q_equal = (10 * mg) / (10 * mg)
    assert isclose(q_equal.value, 1)
    assert f"{q_equal}" == "1"

@pytest.mark.regression(reason="Issue #72: LinearQuantity.__repr__ incorrectly upgrades a cancelled unit to a prefixed SI unit (e.g., mg becomes µkg)")
def test_issue72_repr_cancellation_to_si_symbol(monkeypatch):
    """
    Tests that 'mJ/ms' simplifies to 'W'.
    This confirms the existing behavior still works with the *real* prettifier.
    """
    utils.invalidate_preferred_cache()
    
    mJ = ureg.get("mJ")
    ms = ureg.get("ms")
    
    q_W = (1 * mJ) / (1 * ms) # (1e-3 J) / (1e-3 s) = 1 J/s = 1 W
    
    # Prettify returns 'mJ/ms'
    # __repr__ sees 'mJ/ms', is_composed = True
    # dim is Power, preferred = 'W'
    # scale_to_si is 1.0
    # Logic upgrades to 'W'
    assert f"{q_W}" == "1 W"


# --- Tests for __repr__ Auto-Prefixing of Composed Units ---

@pytest.mark.regression(reason="Issue #75: Fix for non-standard SI scales in __repr__")
def test_repr_upgrades_non_standard_scale_N_per_cm2():
    """
    Tests the original problem: 4 N/cm² (scale 10^4) should be
    auto-formatted to 40 kPa (using engineering notation).
    """
    N = ureg.get("N")
    cm = ureg.get("cm")

    force = 100 * N
    area = 25 * cm**2
    pressure = force / area  # 4 N/cm²

    # Check that the value is correct (4e4 Pa)
    assert pressure.dim == u.Pa.dim
    assert isclose(pressure._mag_si, 40000.0)
    assert pressure.unit.name == "kPa"
    assert isclose(pressure.unit.scale_to_si, 1000.0)

    # Check that __repr__ now correctly formats 4e4 Pa as 40 kPa
    assert f"{pressure}" == "40 kPa"


@pytest.mark.regression(reason="Issue #75: Fix for non-standard SI scales in __repr__")
def test_repr_upgrades_non_standard_scale_kN_per_cm2():
    """
    Tests a more complex non-standard scale: 4 kN/cm² (scale 10^7)
    should be auto-formatted to 40 MPa.
    """
    kN = ureg.get("kN")
    cm = ureg.get("cm")

    force = 100 * kN
    area = 25 * cm**2
    pressure = force / area  # 4 kN/cm²

    # Check that the value is correct (4e7 Pa)
    assert pressure.dim == u.Pa.dim
    assert isclose(pressure._mag_si, 40_000_000.0)
    assert pressure.unit.name == "MPa"
    assert isclose(pressure.unit.scale_to_si, 10**6)

    # Check that __repr__ now correctly formats 4e7 Pa as 40 MPa
    assert f"{pressure}" == "40 MPa"


@pytest.mark.regression(reason="Issue #75: Fix for non-standard SI scales in __repr__")
def test_repr_upgrades_non_standard_scale_small_value():
    """
    Tests a non-standard small scale: 4 N/km² (scale 10^-6)
    This scale *does* match a prefix (micro, µ), so the *old* logic
    path should handle it correctly, formatting it as 4 µPa.
    """
    N = ureg.get("N")
    km = ureg.get("km")

    force = 100 * N
    area = 25 * km**2
    pressure = force / area  # 4 N/km²

    # Check value (4e-6 Pa)
    assert isclose(pressure._mag_si, 4.0e-6)
    assert isclose(pressure.unit.scale_to_si, 1.0e-6)

    # Check that __repr__ formats this as 4 µPa
    # This confirms the pre-existing prefix-matching logic still works
    assert f"{pressure}" == "4 µPa"


# --- Tests to Confirm Regressions Stay Fixed ---

@pytest.mark.regression(reason="Issue #75: Confirm fix for __repr__ bug #72")
def test_repr_regression_fix_issue72_kg_mg_per_kg(monkeypatch):
    """
    Confirms that 'kg·mg/kg' is prettified to 'mg' *before*
    the composed-unit check, preventing it from being upgraded to 'mkg'.
    """
    # This test doesn't need the nop_prettifier, it relies on the real one
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()

    kg = ureg.get("kg")
    mg = ureg.get("mg")

    dose_rate = 15 * (mg / kg)
    patient_mass = 75 * kg
    required_dose = patient_mass * dose_rate  # 0.001125 kg

    assert required_dose.unit.name == "kg"
    assert required_dose.dim == MASS

    # The fix ensures this prints "1125 mg", not "1.125 mkg"
    assert f"{required_dose}" == "0.001125 kg"


@pytest.mark.regression(reason="Issue #75: Confirm fix for __repr__ bug")
def test_repr_regression_fix_keeps_non_si_unit(monkeypatch):
    """
    Confirms that a simple non-SI unit ('cm') is not auto-formatted
    by the new logic.
    """
    _nop_prettifier(monkeypatch) # Use nop to check the raw unit name
    import quantium.core.utils as utils
    monkeypatch.setattr(utils, "preferred_symbol_for_dim", lambda d: "m", raising=True)

    cm = LinearUnit("cm", 0.01, LENGTH)
    q = 2 * cm

    # The fix ensures this prints "2 cm", not "20 mm"
    assert repr(q) == "2 cm"


@pytest.mark.regression(reason="Issue #75: Confirm fix for __repr__ bug")
def test_repr_regression_fix_atomic_units_not_flipped(monkeypatch):
    """
    Confirms that atomic SI symbols (Bq, Hz) are not auto-formatted,
    even though they share a dimension.
    """
    _nop_prettifier(monkeypatch)
    import quantium.core.utils as utils
    utils.invalidate_preferred_cache()
    
    Bq = ureg.get("Bq")
    Hz = ureg.get("Hz")
    q_Bq = 100 * Bq
    q_Hz = 100 * Hz
    
    assert q_Bq.dim == TIME_INV
    assert q_Hz.dim == TIME_INV

    # The fix ensures these print as-is, not flipping to the other
    assert f"{q_Bq}" == "100 Bq"
    assert f"{q_Hz}" == "100 Hz"


@pytest.mark.regression(reason="Issue #75: Confirm fix for __repr__ bug")
def test_repr_regression_fix_format_si_works(monkeypatch):
    """
    Confirms that the ':si' formatter still works, as it relies on
    __repr__ not re-formatting the base SI unit ('Pa').
    """
    _nop_prettifier(monkeypatch)

    kPa = ureg.get("kPa")
    uF = ureg.get("uF")
    
    p = 2 * kPa   # 2000 Pa
    q = 3 * uF    # 3e-6 F

    # p.to_si() creates a LinearQuantity(2000, unit=Pa).
    # The fix ensures repr(LinearQuantity(2000, unit=Pa)) is "2000 Pa",
    # not "2 kPa".
    assert f"{p:si}" == "2000 Pa"

    # q.to_si() creates LinearQuantity(3e-6, unit=F).
    # The fix ensures repr(LinearQuantity(3e-6, unit=F)) is "3e-06 F",
    # not "3 µF".
    assert f"{q:si}" == "3e-06 F"

@pytest.mark.regression(reason="Issue #75: Test __repr__ auto-prefixing for zero values")
def test_repr_handles_zero_value_of_composed_unit(monkeypatch):
    """
    Tests the `if mag_si == 0.0:` branch.
    Ensures that a zero-valued composed unit (like 0 N/cm²) is
    formatted as '0 <BaseSISymbol>' (e.g., '0 Pa').
    """
    _nop_prettifier(monkeypatch) # Not strictly needed, but good for consistency
    N = ureg.get("N")
    cm = ureg.get("cm")

    # Create a zero-value quantity with a composed unit
    pressure = 0 * (N / cm**2)

    assert isclose(pressure._mag_si, 0.0)
    assert pressure.unit.name == "Pa"
    assert pressure.dim == u.Pa.dim

    # The logic should catch mag_si == 0.0 and print '0 Pa'
    assert f"{pressure}" == "0 Pa"

    # Test another zero-value quantity
    force = 0 * (ureg.get("kg") * ureg.get("m") / ureg.get("ms")**2)
    assert isclose(force._mag_si, 0.0)
    assert f"{force}" == "0 N"


@pytest.mark.regression(reason="Issue #75: Test __repr__ auto-prefixing for base-range values")
def test_repr_handles_base_unit_range_value(monkeypatch):
    """
    Tests the `if prefix_exp == 0:` branch.
    Ensures that a composed unit whose SI value is in the base
    range (e.g., 4.0 N/m² = 4.0 Pa) is correctly formatted
    with the base SI symbol (e.g., '4 Pa').
    """
    _nop_prettifier(monkeypatch)
    N = ureg.get("N")
    m = ureg.get("m")

    # This is the test case from your original example
    force = 100 * N
    area = 25 * m**2
    pressure = force / area  

    assert isclose(pressure._mag_si, 4.0)
    assert pressure.unit.name == "Pa"
    assert pressure.dim == u.Pa.dim

    # The logic should find mag_si = 4.0, calculate prefix_exp = 0,
    # and correctly format it as '4 Pa'.
    assert f"{pressure}" == "4 Pa"


def test_unit_format_rules_all(monkeypatch):
    _nop_prettifier(monkeypatch)
    utils.invalidate_preferred_cache()

    m = ureg.get("m")
    cm = ureg.get("cm")

    q_mul = (1 * m) * ((1 * cm) ** 2)
    assert isclose(q_mul.value, 100.0)
    assert q_mul.unit.name == "cm^3"

    q_mul_rev = ((1 * cm) ** 2) * (1 * m)
    assert isclose(q_mul_rev.value, 100.0)
    assert q_mul_rev.unit.name == "cm^3"

    q_div = (1 * m) / ((1 * cm) ** 2)
    assert isclose(q_div.value, 100.0)
    assert q_div.unit.name == "1/cm"

    q_div_rev = ((1 * cm) ** 2) / (1 * m)
    assert isclose(q_div_rev.value, 0.01)
    assert q_div_rev.unit.name == "cm"

    q_tie_cm_first = (1 * cm) * (1 * m)
    assert isclose(q_tie_cm_first.value, 100.0)
    assert q_tie_cm_first.unit.name == "cm^2"

    q_tie_m_first = (1 * m) * (1 * cm)
    assert isclose(q_tie_m_first.value, 0.01)
    assert q_tie_m_first.unit.name == "m^2"

    C = ureg.get("C")
    uC = ureg.get("µC")
    s = ureg.get("s")
    ms = ureg.get("ms")

    q_A = (1 * C) / (1 * s)
    assert isclose(q_A.value, 1.0)
    assert q_A.unit.name == "A"

    q_mA = (1 * uC) / (1 * ms)
    assert isclose(q_mA.value, 1.0)
    assert q_mA.unit.name == "mA"

    J = ureg.get("J")
    kJ = ureg.get("kJ")

    q_W = (1 * J) / (1 * s)
    assert isclose(q_W.value, 1.0)
    assert q_W.unit.name == "W"

    q_MW = (1 * kJ) / (1 * ms)
    assert isclose(q_MW.value, 1.0)
    assert q_MW.unit.name == "MW"

    N = ureg.get("N")

    q_kPa = (1 * N) / ((1 * cm) ** 2)
    assert isclose(q_kPa.value, 10.0)
    assert q_kPa.unit.name == "kPa"

    Pa = ureg.get("Pa")
    kPa = ureg.get("kPa")

    q_tie_pa = (1 * Pa) * (1 * kPa)
    assert isclose(q_tie_pa.value, 1000.0)
    assert q_tie_pa.unit.name == "Pa^2"

    q_tie_kpa = (1 * kPa) * (1 * Pa)
    assert isclose(q_tie_kpa.value, 0.001)
    assert q_tie_kpa.unit.name == "kPa^2"

    kN = ureg.get("kN")

    q_force_power = ((1 * kN) ** 2) * (1 * N)
    assert isclose(q_force_power.value, 0.001)
    assert q_force_power.unit.name == "kN^3"

    q_force_power_rev = (1 * N) * ((1 * kN) ** 2)
    assert isclose(q_force_power_rev.value, 0.001)
    assert q_force_power_rev.unit.name == "kN^3"

    q_energy_div = ((1 * J) ** 2) / ((1 * kJ) ** 3)
    assert isclose(q_energy_div.value, 1e-6)
    assert q_energy_div.unit.name == "1/kJ"

    q_energy_mul = ((1 * J) ** 2) * ((1 * kJ) ** 3)
    assert isclose(q_energy_mul.value, 1e-6)
    assert q_energy_mul.unit.name == "kJ^5"
