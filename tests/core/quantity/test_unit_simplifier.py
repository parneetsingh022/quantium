import pytest

from quantium.core.dimensions import DIM_0, TIME, LENGTH, MASS, CURRENT, LUMINOUS, dim_div, dim_mul, dim_pow
from quantium.core.unit import LinearUnit
from quantium.io.unit_simplifier import UnitNameSimplifier
from quantium.units.registry import DEFAULT_REGISTRY
from fractions import Fraction


UNIT_SIMPLIFIER = UnitNameSimplifier(LinearUnit)

@pytest.fixture(scope="module")
def simplifier() -> UnitNameSimplifier:
    return UnitNameSimplifier(LinearUnit)


def test_canonical_unit_for_dim_prefers_registered_symbol(simplifier: UnitNameSimplifier):
    newton = DEFAULT_REGISTRY.get("N")
    canonical = simplifier.canonical_unit_for_dim(newton.dim)
    assert canonical.name == "N"
    assert canonical.scale_to_si == pytest.approx(1.0)
    assert canonical.dim == newton.dim


def test_canonical_unit_for_dim_detects_preferred_powers(simplifier: UnitNameSimplifier):
    joule = DEFAULT_REGISTRY.get("J")
    squared_dim = dim_pow(joule.dim, 2)
    canonical = simplifier.canonical_unit_for_dim(squared_dim)
    assert canonical.name == "J^2"
    assert canonical.scale_to_si == pytest.approx(1.0)
    assert canonical.dim == squared_dim


def test_normalize_power_name():
    normalize = UnitNameSimplifier.normalize_power_name
    assert normalize("m^1") == "m"
    assert normalize("m^0") == "1"
    assert normalize("m^-2") == "m^-2"
    assert normalize("Pa") == "Pa"


def test_unit_symbol_map_records_priority_and_index(simplifier: UnitNameSimplifier):
    metre = DEFAULT_REGISTRY.get("m")
    mapping = simplifier.unit_symbol_map(metre, priority=2)
    assert mapping == {"m": (1, (2, 0))}


def test_scale_symbol_map_scales_and_drops_zero(simplifier: UnitNameSimplifier):
    mapping = {"m": (1, (0, 0)), "s": (0, (1, 0))}
    scaled = simplifier.scale_symbol_map(mapping, 2)
    assert scaled["m"] == (2, (0, 0))
    assert "s" not in scaled


def test_combine_symbol_maps_merges_and_cancels(simplifier: UnitNameSimplifier):
    map_a = {"m": (1, (0, 0))}
    map_b = {"m": (-1, (1, 0)), "s": (2, (1, 1))}
    combined = simplifier.combine_symbol_maps(map_a, map_b)
    assert "m" not in combined
    assert combined["s"] == (2, (1, 1))


def test_format_unit_components_handles_fraction(simplifier: UnitNameSimplifier):
    components = {"m": (1, (0, 0)), "s": (-2, (1, 0))}
    assert simplifier.format_unit_components(components) == "m/s^2"


def test_unit_from_components_rebuilds_registered_units(simplifier: UnitNameSimplifier):
    newton = DEFAULT_REGISTRY.get("N")
    metre = DEFAULT_REGISTRY.get("m")
    components = simplifier.combine_symbol_maps(
        simplifier.unit_symbol_map(newton, 0),
        simplifier.unit_symbol_map(metre, 1),
    )
    composite = simplifier._unit_from_components(components)
    expected = newton * metre
    assert composite == expected


def test_si_to_value_unit_dimensionless_returns_identity(simplifier: UnitNameSimplifier):
    value, unit = simplifier.si_to_value_unit(5.0, DIM_0)
    assert value == pytest.approx(5.0)
    assert unit.name == ""
    assert unit.dim == DIM_0


def test_si_to_value_unit_preserves_requested_simple_unit(simplifier: UnitNameSimplifier):
    centimetre = DEFAULT_REGISTRY.get("cm")
    components = simplifier.unit_symbol_map(centimetre)
    value, unit = simplifier.si_to_value_unit(0.05, centimetre.dim, components, centimetre)
    assert value == pytest.approx(5.0)
    assert unit.name == "cm"


def test_si_to_value_unit_prefers_prefix_for_readable_values(simplifier: UnitNameSimplifier):
    metre = DEFAULT_REGISTRY.get("m")
    components = simplifier.unit_symbol_map(metre)
    value, unit = simplifier.si_to_value_unit(0.001, metre.dim, components)
    assert unit.name == "mm"
    assert value == pytest.approx(1.0)


def test_si_to_value_unit_prefers_highest_exponent_component(simplifier: UnitNameSimplifier):
    newton = DEFAULT_REGISTRY.get("N")
    kilonewton = DEFAULT_REGISTRY.get("kN")
    kilonewton_sq = kilonewton ** 2

    components = simplifier.combine_symbol_maps(
        simplifier.unit_symbol_map(newton, 0),
        simplifier.unit_symbol_map(kilonewton_sq, 1),
    )

    combined_dim = dim_mul(newton.dim, kilonewton_sq.dim)
    combined_scale = newton.scale_to_si * kilonewton_sq.scale_to_si

    value, unit = simplifier.si_to_value_unit(combined_scale, combined_dim, components)

    expected_unit = kilonewton ** 3
    assert unit.name == expected_unit.name
    assert unit.dim == expected_unit.dim
    assert value == pytest.approx(combined_scale / expected_unit.scale_to_si)


# Each case: (case_name, target_dim_fn, components_dict)
# components_dict maps symbol -> (Fraction exponent, (priority,index))
CASES = [
    # Speed: m/s
    ("speed", lambda: dim_div(LENGTH, TIME),
     {"m": (Fraction(1, 1), (0, 0)), "s": (Fraction(-1, 1), (0, 1))}),

    # Acceleration: m/s^2
    ("acceleration", lambda: dim_div(LENGTH, dim_pow(TIME, 2)),
     {"m": (Fraction(1, 1), (0, 0)), "s": (Fraction(-2, 1), (0, 1))}),

    # Pressure: N/m^2
    ("pressure", lambda: dim_div(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), dim_pow(LENGTH, 2)),
     {"N": (Fraction(1, 1), (0, 0)), "m": (Fraction(-2, 1), (0, 1))}),

    # Energy: N·m
    ("energy", lambda: dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH),
     {"N": (Fraction(1, 1), (0, 0)), "m": (Fraction(1, 1), (0, 1))}),

    # Power: J/s
    ("power", lambda: dim_div(
        dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME),
     {"J": (Fraction(1, 1), (0, 0)), "s": (Fraction(-1, 1), (0, 1))}),

    # Voltage: W/A
    ("voltage", lambda: dim_div(
        dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT),
     {"W": (Fraction(1, 1), (0, 0)), "A": (Fraction(-1, 1), (0, 1))}),

    # Capacitance: C/V
    ("capacitance", lambda: dim_div(
        dim_mul(CURRENT, TIME),
        dim_div(
            dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT
        )),
     {"C": (Fraction(1, 1), (0, 0)), "V": (Fraction(-1, 1), (0, 1))}),

    # Resistance: V/A
    ("resistance", lambda: dim_div(
        dim_div(
            dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT
        ), CURRENT),
     {"V": (Fraction(1, 1), (0, 0)), "A": (Fraction(-1, 1), (0, 1))}),

    # Magnetic flux: V·s (Wb)
    ("flux", lambda: dim_mul(
        dim_div(
            dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT
        ), TIME),
     {"V": (Fraction(1, 1), (0, 0)), "s": (Fraction(1, 1), (0, 1))}),

    # Flux density: Wb/m^2 (T)
    ("flux_density", lambda: dim_div(
        dim_mul(
            dim_div(
                dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT
            ), TIME),
        dim_pow(LENGTH, 2)),
     {"Wb": (Fraction(1, 1), (0, 0)), "m": (Fraction(-2, 1), (0, 1))}),

    # Inductance: Wb/A (H)
    ("inductance", lambda: dim_div(
        dim_mul(
            dim_div(
                dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT
            ), TIME),
        CURRENT),
     {"Wb": (Fraction(1, 1), (0, 0)), "A": (Fraction(-1, 1), (0, 1))}),

    # Absorbed dose: J/kg (Gy)
    ("dose", lambda: dim_div(
        dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), MASS),
     {"J": (Fraction(1, 1), (0, 0)), "kg": (Fraction(-1, 1), (0, 1))}),

    # Frequency: 1/s (Hz, Bq)
    ("frequency", lambda: dim_pow(TIME, -1),
     {"s": (Fraction(-1, 1), (0, 0))}),

    # Illuminance: lm/m^2 (lx)  — luminous is dimensionless “sr” factor already folded
    ("illuminance", lambda: dim_div(LUMINOUS, dim_pow(LENGTH, 2)),
     {"lm": (Fraction(1, 1), (0, 0)), "m": (Fraction(-2, 1), (0, 1))}),
]


@pytest.mark.parametrize("name,dim_fn,components", CASES, ids=[c[0] for c in CASES])
def test_prefixed_candidates_base_none_keeps_composite_many(monkeypatch, name, dim_fn, components):
    """
    For a variety of derived dimensions, force preferred_symbol_for_dim(dim) to an
    unknown head ('zz') so _prefixed_candidates_for_symbol('zz') returns [].
    The simplifier must then keep the composite built from the provided components.
    """
    from quantium.core import utils as utils_mod
    original = utils_mod.preferred_symbol_for_dim

    target_dim = dim_fn()

    def fake_preferred_symbol_for_dim(dim):
        if dim == target_dim:
            return "zz"  # not registered → base=None → []
        return original(dim)

    monkeypatch.setattr(utils_mod, "preferred_symbol_for_dim",
                        fake_preferred_symbol_for_dim, raising=True)

    mag_si = 1.0  # pick 1.0 SI magnitude for easy assertions
    value, unit = UNIT_SIMPLIFIER.si_to_value_unit(
        mag_si, target_dim, components, requested_unit=None
    )

    # We don’t pin the display name; we just assert physics stayed intact.
    assert unit.dim == target_dim
    # Composite we built uses only SI heads, so scale_to_si should be 1
    assert unit.scale_to_si == pytest.approx(1.0)
    assert value == pytest.approx(1.0)

def test_collapse_same_dim_symbol_unknown_returns_none_then_canonical_fallback():
    """
    Provide an unknown component symbol so _registry_get(sym) resolves to None,
    triggering the early 'return None' from _collapse_same_dim_components.
    That prevents a collapse and composite build; we then fall back to the
    canonical unit for the dimension (length -> 'm').
    """
    dim_length = (1, 0, 0, 0, 0, 0, 0)
    mag_si = 2.0  # 2 meters

    comps = {
        "totally_unknown_symbol": (Fraction(1, 1), (0, 0)),
    }

    value, unit = UNIT_SIMPLIFIER.si_to_value_unit(
        mag_si, dim_length, comps, requested_unit=None
    )

    # Canonical fallback for length should be meters in your setup
    assert unit.dim == dim_length
    assert unit.name in {"m"}  # preferred length symbol
    assert value == pytest.approx(2.0)




def test_collapse_same_dim_dim_mismatch_falls_through_final_none():
    """
    Provide only length components whose exponents sum to 2 (e.g., m * cm),
    but set target dim to L*T. The collapse forms L^2, which doesn't match
    the target, so _collapse_same_dim_components hits the final 'return None'.
    The overall algorithm then proceeds and ultimately returns a unit
    matching L*T (e.g., something equivalent to m*s).
    """
    comps = {
        "m":  (Fraction(1, 1), (0, 0)),
        "cm": (Fraction(1, 1), (0, 1)),
    }

    dim_L_times_T = (1, 0, 1, 0, 0, 0, 0)  # L^1 * T^1
    mag_si = 1.0

    value, unit = UNIT_SIMPLIFIER.si_to_value_unit(
        mag_si, dim_L_times_T, comps, requested_unit=None
    )

    assert unit.dim == dim_L_times_T
    # Value should be finite; exact symbol may vary (e.g., 'm·s')
    assert value == pytest.approx(1.0)
    assert unit.name == "m·s"


def test_preferred_power_collapse_positive_power_from_components():
    """
    Target L^2 (area). The preferred base for LENGTH is 'm', and we include 'm'
    in components so the '_match_preferred_power' branch triggers and returns
    'm^2' directly (instead of keeping a composite).
    """
    dim_area = dim_pow(LENGTH, 2)

    # Components explicitly reference the base symbol 'm'
    comps = {
        "m": (Fraction(2, 1), (0, 0)),  # m^2
    }

    mag_si = 12.5  # 12.5 m^2 in SI
    value, unit = UNIT_SIMPLIFIER.si_to_value_unit(
        mag_si, dim_area, comps, requested_unit=None
    )

    assert unit.dim == dim_area
    # Name normalization for LinearUnit ** 2 should yield an 'm^2'-like form
    assert unit.scale_to_si == pytest.approx(1.0)
    assert value == pytest.approx(12.5)
    # Keep this loose since your formatter may use superscripts; but the base should be 'm'
    assert "m" in unit.name and ("^2" in unit.name or "²" in unit.name)


def test_preferred_power_collapse_negative_power_from_components():
    """
    Target T^-3 (e.g., 1/s^3). The preferred base for TIME is 's', and we include 's'
    in components so the branch returns 's^-3' (reciprocal cubic seconds).
    """
    dim_inv_time_cubed = dim_pow(TIME, -3)

    # Components explicitly reference the base symbol 's'
    comps = {
        "s": (Fraction(-3, 1), (0, 0)),  # s^-3 = 1/s^3
    }

    mag_si = 8.0  # 8 (1/s^3) in SI
    value, unit = UNIT_SIMPLIFIER.si_to_value_unit(
        mag_si, dim_inv_time_cubed, comps, requested_unit=None
    )

    assert unit.dim == dim_inv_time_cubed
    assert unit.scale_to_si == pytest.approx(1.0)
    assert value == pytest.approx(8.0)
    # Again, don't over-specify formatting; ensure base 's' and a power of 3 are present
    assert "s" in unit.name
    assert ("^3" in unit.name) or ("3" in unit.name) or ("³" in unit.name)  # tolerant to pretty format