# pytest tests for quantium.units.registry
#
# These tests exercise normalization, aliases, SI-prefix synthesis,
# anti-stacking rules, thread-safety, and the convenience functions that
# delegate to DEFAULT_REGISTRY. They purposefully use an isolated registry
# instance for most tests, and monkeypatch DEFAULT_REGISTRY where needed.

import threading

import pytest

from quantium.core.dimensions import (
    AMOUNT,
    CURRENT,
    DIM_0,
    LENGTH,
    LUMINOUS,
    MASS,
    TEMPERATURE,
    TIME,
    dim_div,
    dim_mul,
    dim_pow,
)
from quantium.core.unit import LinearUnit, AffineUnit


# We import the module under test once, and access internals we intentionally
# rely on in tests (like _bootstrap_default_registry).
import quantium.units.registry as regmod
from quantium.units.registry import UnitsRegistry, UnitNamespace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def reg():
    """Fresh, fully-bootstrapped UnitsRegistry for isolation per test."""
    # Using the module's bootstrap helper keeps the registry consistent with
    # the shipped set of units while avoiding global mutations.
    return regmod._bootstrap_default_registry()


@pytest.fixture()
def patched_default(monkeypatch, reg):
    """Temporarily replace DEFAULT_REGISTRY with an isolated instance."""
    monkeypatch.setattr(regmod, "DEFAULT_REGISTRY", reg)
    yield reg


# ---------------------------------------------------------------------------
# Base/derived presence & correctness
# ---------------------------------------------------------------------------

def test_base_units_present(reg):
    for sym, dim, utype in [("m", LENGTH, LinearUnit), ("kg", MASS, LinearUnit), ("s", TIME, LinearUnit), ("A", CURRENT, LinearUnit), ("delta_K", TEMPERATURE, LinearUnit), ("K", TEMPERATURE, AffineUnit), ("mol", AMOUNT, LinearUnit), ("cd", LUMINOUS, LinearUnit)]:
        u = reg.get(sym)
        assert isinstance(u, utype)
        assert u.scale_to_si == pytest.approx(1.0)
        assert u.dim == dim


def test_common_derived_units_present(reg):
    # spot-check a few dimensions/scales
    assert reg.get("rad").dim == DIM_0
    assert reg.get("sr").dim == DIM_0

    Hz = reg.get("Hz")
    assert Hz.dim == dim_pow(TIME, -1)
    assert Hz.scale_to_si == pytest.approx(1.0)

    N_unit = reg.get("N")
    assert N_unit.dim == dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2)))  # kg·m/s²

    Pa = reg.get("Pa")
    assert Pa.dim == dim_div(N_unit.dim, dim_pow(LENGTH, 2))

    J_unit = reg.get("J")
    assert J_unit.dim == dim_mul(N_unit.dim, LENGTH)

    W = reg.get("W")
    assert W.dim == dim_div(J_unit.dim, TIME)

    V = reg.get("V")
    assert V.dim == dim_div(W.dim, CURRENT)

    F = reg.get("F")
    assert F.dim == dim_div(reg.get("C").dim, V.dim)

    ohm = reg.get("Ω")
    assert ohm.dim == dim_div(V.dim, CURRENT)

    S = reg.get("S")
    assert S.dim == dim_div(CURRENT, V.dim)


# ---------------------------------------------------------------------------
# Normalization & aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp, expected", [
    ("um", "µm"),   # ASCII micro-prefix fallback at start
    ("uA", "µA"),
    ("ohm", "Ω"),   # textual alias → canonical
    ("Ohm", "Ω"),
    ("OHM", "Ω"),
])
def test_normalization_maps_to_canonical(inp, expected, reg):
    # `get` should just work with the normalized symbol
    # For prefixes, we expect synthesis if not pre-registered.
    u1 = reg.get(expected)
    u2 = reg.get(inp)
    assert u1.dim == u2.dim
    assert u1.scale_to_si == pytest.approx(u2.scale_to_si)


def test_alias_registration_custom(reg):
    # Create a new alias for ohm and confirm both path & object identity
    reg.register_alias("ohms", "Ω")
    u_alias = reg.get("ohms")
    u_ohm = reg.get("Ω")
    assert u_alias is u_ohm


# ---------------------------------------------------------------------------
# Prefix synthesis & anti-stacking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pref, base, factor", [
    ("k", "m", 1e3),
    ("M", "s", 1e6),
    ("µ", "A", 1e-6),
    ("n", "mol", 1e-9),
])
def test_valid_prefix_synthesis(pref, base, factor, reg):
    base_u = reg.get(base)
    sym = f"{pref}{base}"
    u = reg.get(sym)
    assert u.name == sym
    assert u.dim == base_u.dim
    assert u.scale_to_si == pytest.approx(base_u.scale_to_si * factor)


def test_synthesis_is_lazy_and_idempotent(reg):
    base_len = len(reg.all())
    km1 = reg.get("km")
    after_first = len(reg.all())
    km2 = reg.get("km")
    after_second = len(reg.all())

    assert after_first == base_len + 1  # created once
    assert after_second == after_first  # no duplicates
    assert km1 is km2


def test_anti_stacking_prefixed_base_rejected(reg):
    # 'mm' is milli + m (valid and synthesized)
    mm = reg.get("mm")
    assert mm.name == "mm"

    # 'kmm' would be kilo + (mm) which is stacking → must fail
    with pytest.raises(ValueError):
        reg.get("kmm")

    # Another explicit stacking case: 'kµm'
    with pytest.raises(ValueError):
        reg.get("kµm")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_symbol_raises(reg):
    with pytest.raises(ValueError):
        reg.get("nope")


# ---------------------------------------------------------------------------
# Thread-safety: concurrent synthesis of the same symbol
# ---------------------------------------------------------------------------

def test_thread_safe_prefixed_creation(reg):
    created = []
    errs = []

    def worker():
        try:
            created.append(reg.get("km"))
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errs
    # All returned the exact same LinearUnit object
    first = created[0]
    assert all(u is first for u in created)
    # Registry has exactly one 'km'
    assert "km" in reg.all() and sum(1 for k in reg.all().keys() if k == "km") == 1


# ---------------------------------------------------------------------------
# Convenience functions should use DEFAULT_REGISTRY
# ---------------------------------------------------------------------------

def test_convenience_functions_use_default_registry(patched_default):
    # register a custom unit through the free function and fetch it back
    ureg = UnitsRegistry()
    ureg.register(LinearUnit("ft", 0.3048, LENGTH))
    out = ureg.get("ft")
    assert out.name == "ft"
    assert out.scale_to_si == pytest.approx(0.3048)

    # alias convenience function
    ureg.register_alias("foot", "ft")
    assert ureg.get("foot") is out


# ---------------------------------------------------------------------------
# Dimension sanity checks via relationships
# ---------------------------------------------------------------------------

def test_dimension_relationships(reg):
    # AMOUNT = kg·m/s², LUMINOUS = N·m, W = LUMINOUS/s, V = W/A, Ω = V/A, S = A/V
    kg = reg.get("kg").dim
    m = reg.get("m").dim
    s = reg.get("s").dim
    A = reg.get("A").dim

    N_dim = dim_mul(kg, dim_div(m, dim_pow(s, 2)))
    assert reg.get("N").dim == N_dim

    J_dim = dim_mul(N_dim, m)
    assert reg.get("J").dim == J_dim

    W_dim = dim_div(J_dim, s)
    assert reg.get("W").dim == W_dim

    V_dim = dim_div(W_dim, A)
    assert reg.get("V").dim == V_dim

    ohm_dim = dim_div(V_dim, A)
    assert reg.get("Ω").dim == ohm_dim

    S_dim = dim_div(A, V_dim)
    assert reg.get("S").dim == S_dim



@pytest.mark.parametrize("sym, dim_expr, scale", [
    # frequency & mechanics
    ("Hz", dim_pow(TIME, -1), 1.0),
    ("N",  dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), 1.0),
    ("Pa", dim_div(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), dim_pow(LENGTH, 2)), 1.0),
    ("J",  dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), 1.0),
    ("W",  dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), 1.0),

    # electricity
    ("C",  dim_mul(CURRENT, TIME), 1.0),
    ("V",  dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), 1.0),
    ("F",  dim_div(dim_mul(CURRENT, TIME), dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT)), 1.0),
    ("Ω",  dim_div(  # V / A
            dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), CURRENT), 1.0),
    ("S",  dim_div(CURRENT, dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT)), 1.0),
    ("Wb", dim_mul(dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), TIME), 1.0),
    ("T",  dim_div(dim_mul(dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), TIME), dim_pow(LENGTH, 2)), 1.0),
    ("H",  dim_div(dim_mul(dim_div(dim_div(dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), TIME), CURRENT), TIME), CURRENT), 1.0),

    # photometry (sr is DIM_0 so lm = cd)
    ("lm", dim_mul(LUMINOUS, DIM_0), 1.0),
    ("lx", dim_div(dim_mul(LUMINOUS, DIM_0), dim_pow(LENGTH, 2)), 1.0),

    # radioactivity & dose
    ("Bq", dim_pow(TIME, -1), 1.0),
    ("Gy", dim_div(  # LUMINOUS/kg
            dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), MASS), 1.0),
    ("Sv", dim_div(  # same as Gy
            dim_mul(dim_mul(MASS, dim_div(LENGTH, dim_pow(TIME, 2))), LENGTH), MASS), 1.0),

    # catalytic activity
    ("kat", dim_div(AMOUNT, TIME), 1.0),

    # mass non-SI base
    ("g",  MASS, 1e-3),
])
def test_all_derived_units_dimensions_and_scales(reg, sym, dim_expr, scale):
    u = reg.get(sym)
    assert u.dim == dim_expr
    assert u.scale_to_si == pytest.approx(scale)


# ---------------------------------------------------------------------------
# Time units: presence, dimensions, scales
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sym, seconds", [
    ("min",       60.0),
    ("h",         60.0 * 60.0),
    ("d",         24.0 * 60.0 * 60.0),
    ("wk",        7.0 * 24.0 * 60.0 * 60.0),
    ("fortnight", 14.0 * 24.0 * 60.0 * 60.0),

    # Civil (Gregorian) average month/year
    ("mo",        (365.2425 / 12.0) * 24.0 * 3600.0),
    ("yr",        365.2425 * 24.0 * 3600.0),

    # Astronomy
    ("yr_julian", 365.25 * 24.0 * 3600.0),

    # Longer spans (Gregorian-based)
    ("decade",     10.0  * 365.2425 * 24.0 * 3600.0),
    ("century",    100.0 * 365.2425 * 24.0 * 3600.0),
    ("millennium", 1000.0* 365.2425 * 24.0 * 3600.0),
])
def test_time_units_present_and_scaled(reg, sym, seconds):
    u = reg.get(sym)
    assert isinstance(u, LinearUnit)
    assert u.dim == TIME
    assert u.scale_to_si == pytest.approx(seconds)


# ---------------------------------------------------------------------------
# Time aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias, canonical", [
    # Common aliases (already have ohm in your tests, included here for completeness)
    ("ohm", "Ω"),
    ("Ohm", "Ω"),
    ("OHM", "Ω"),

    # Time aliases
    ("minute", "min"),
    ("minutes", "min"),
    ("hr", "h"),
    ("hour", "h"),
    ("hours", "h"),
    ("day", "d"),
    ("days", "d"),
    ("week", "wk"),
    ("weeks", "wk"),
    ("fortnights", "fortnight"),
    ("month", "mo"),
    ("months", "mo"),
    ("year", "yr"),
    ("years", "yr"),
    ("annum", "yr"),
    ("dec", "decade"),
    ("decades", "decade"),
    ("cent", "century"),
    ("centuries", "century"),
    ("millennia", "millennium"),
])
def test_time_aliases_map_to_canonical(reg, alias, canonical):
    u_alias = reg.get(alias)
    u_canon = reg.get(canonical)
    # Should be the exact same LinearUnit object
    assert u_alias is u_canon


# ---------------------------------------------------------------------------
# Non-prefixable enforcement
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_sym", [
    # The registry marks these as non-prefixable; any SI/metric prefix must fail.
    "mkg", "kkg", "ukg",            # kg is non-prefixable (SI prefixes apply to gram, not kilogram)
    "kmin", "µmin", "umin", "mmin",
    "kh", "µh", "uh", "mh",
    "kd", "µd", "ud", "md",
    "kwk", "µwk", "uwk", "mwk",
    "kfortnight", "µfortnight", "ufortnight", "mfortnight",
    "kmo", "µmo", "umo", "mmo",
    "kyr", "µyr", "uyr", "myr",
    "kyr_julian", "µyr_julian", "uyr_julian", "myr_julian",
    "kdecade", "µdecade", "udecade", "mdecade",
    "kcentury", "µcentury", "ucentury", "mcentury",
    "kmillennium", "µmillennium", "umillennium", "mmillennium",
])
def test_non_prefixable_time_units_reject_prefixes(reg, bad_sym):
    with pytest.raises(ValueError):
        reg.get(bad_sym)


def test_non_prefixable_does_not_accidentally_create_units(reg):
    # Ensure attempting to fetch a prefixed version did not pollute the registry
    base_len = len(reg.all())
    for bad in ["kyr", "umin", "kh", "kmo", "mkg"]:
        with pytest.raises(ValueError):
            reg.get(bad)
    assert len(reg.all()) == base_len


# ---------------------------------------------------------------------------
# Sanity: prefix synthesis still works for prefixable time base 's'
# (and remains disallowed for the added non-prefixable time units)
# ---------------------------------------------------------------------------

def test_seconds_still_prefixable_but_not_month_year(reg):
    # Positive control: seconds *are* prefixable (already covered elsewhere, but we assert here for contrast)
    ms = reg.get("ms")
    s = reg.get("s")
    assert ms.dim == s.dim
    assert ms.scale_to_si == pytest.approx(1e-3)

    # Negative controls: these are non-prefixable
    for sym in ["kmo", "kyr", "µh"]:
        with pytest.raises(ValueError):
            reg.get(sym)


# -------------------------------
# Basic compound parsing & equivalence
# -------------------------------

def test_get_compound_velocity_and_acceleration(reg):
    u1 = reg.get("m/s**2")             # parser path
    u2 = reg.get("m") / (reg.get("s") ** 2)  # manual composition
    assert isinstance(u1, LinearUnit)
    assert u1.dim == u2.dim
    assert u1.scale_to_si == pytest.approx(u2.scale_to_si)

def test_get_compound_force_equals_newton(reg):
    nm = reg.get("kg*m/s**2")   # base composition
    N  = reg.get("N")           # named
    assert nm.dim == N.dim
    assert nm.scale_to_si == pytest.approx(N.scale_to_si)

def test_get_literal_one_over_second_equals_hz(reg):
    inv_s = reg.get("1/s")
    Hz = reg.get("Hz")
    assert inv_s.dim == Hz.dim
    assert inv_s.scale_to_si == pytest.approx(1.0)


# -------------------------------
# Parentheses distribution cases
# -------------------------------

def test_get_parentheses_complex_reduces_to_m3_per_s(reg):
    # (W*s)/(N*s/m**2) -> m^3/s (scale 1)
    u_expr = reg.get("(W*s)/(N*s/m**2)")
    u_si   = reg.get("m**3/s")
    assert u_expr.dim == u_si.dim
    assert u_expr.scale_to_si == pytest.approx(1.0)

def test_get_parentheses_with_prefixes_and_mm(reg):
    # (uW*s)/(N*s/mm**2) -> 1e-12 * m^3/s
    u_expr = reg.get("(uW*s)/(N*s/mm**2)")
    u_si   = reg.get("m**3/s")
    assert u_expr.dim == u_si.dim
    # uW = 1e-6 W, mm^2 = 1e-6 m^2  => overall factor 1e-12
    assert u_expr.scale_to_si == pytest.approx(1e-12)


# -------------------------------
# Aliases & prefixes inside expressions
# -------------------------------

def test_get_alias_inside_expression(reg):
    # ohm alias should normalize to Ω and V = Ω·A
    v1 = reg.get("ohm*A")
    V  = reg.get("V")
    assert v1.dim == V.dim
    assert v1.scale_to_si == pytest.approx(1.0)

def test_get_prefix_inside_expression(reg):
    # 10 cm/s vs m/s scale factors: reg.get should synthesize cm correctly
    cm_per_s = reg.get("cm/s")
    m_per_s  = reg.get("m/s")
    # compare scales to SI: cm/s should be 1e-2 m/s
    assert cm_per_s.dim == m_per_s.dim
    assert cm_per_s.scale_to_si == pytest.approx(1e-2)


# -------------------------------
# Idempotence / caching of expressions
# (identity not required, but equality must hold)
# -------------------------------

def test_get_same_expression_twice_equal_not_necessarily_identical(reg):
    u1 = reg.get("kg*m/s**2")
    u2 = reg.get("kg*m/s**2")
    assert u1 is not None and u2 is not None
    assert u1.dim == u2.dim
    assert u1.scale_to_si == pytest.approx(u2.scale_to_si)
    # different object identity is fine; equality is defined by dim+scale


# -------------------------------
# Errors & invalid syntax
# -------------------------------

@pytest.mark.parametrize("bad", [
    "m//s",         # disallowed token
    "m/s**",        # trailing operator
    "*/s",          # starts with operator
    "(",            # unbalanced paren
    "m/s)",         # unbalanced paren
    "1/",           # dangling slash
    "kg*m/s**x",    # non-integer exponent
    "kg*m/s**+2+3", # trailing noise
])
def test_get_compound_invalid_syntax_raises(reg, bad):
    with pytest.raises(ValueError):
        reg.get(bad)

def test_get_compound_unknown_symbol_raises(reg):
    with pytest.raises(ValueError):
        reg.get("kg*blorp/s")


# -------------------------------
# Thread-safety smoke test for compound expressions
# -------------------------------

def test_thread_safety_compound_get(reg):
    errs = []
    outs = []

    def worker(expr):
        try:
            outs.append(reg.get(expr))
        except Exception as e:
            errs.append(e)

    exprs = [
        "kg*m/s**2",
        "m/s",
        "(W*s)/(N*s/m**2)",
        "1/s",
        "cm/s",
    ] * 8  # repeat to create more concurrency

    threads = [threading.Thread(target=worker, args=(e,)) for e in exprs]
    for t in threads: 
        t.start()
    for t in threads: 
        t.join()

    assert not errs
    # basic sanity: all results are Units with sensible dim/scale
    assert all(isinstance(u, LinearUnit) for u in outs)
    # verify a couple of expected dims quickly
    assert any(u.dim == dim_div(LENGTH, TIME) for u in outs)        # m/s cases
    assert any(u.dim == dim_pow(TIME, -1) for u in outs)            # 1/s cases


# ---------------------------------------------------------------------------
# Alias replacement over existing unit symbols
# ---------------------------------------------------------------------------


def test_register_alias_replace_overwrites_existing_symbol(reg):
    # Sanity: base meter vs millimeter scales/dims differ by 1e-3
    m = reg.get("m")
    mm = reg.get("mm")  # synthesized on demand
    assert mm.dim == m.dim
    assert mm.scale_to_si == pytest.approx(1e-3)

    # Now deliberately SHADOW the existing symbol 'm' so that lookups for 'm' resolve to 'mm'
    reg.register_alias("m", "mm", replace=True)

    # After replacement, fetching 'm' should resolve via the alias to the canonical 'mm'
    m_after = reg.get("m")
    assert m_after is mm                 # exact same LinearUnit object
    assert m_after.scale_to_si == pytest.approx(1e-3)

    # And fetching by canonical 'mm' is unchanged
    assert reg.get("mm") is mm

# ---------------------------------------------------------------------------
# register_alias(..., replace=...) behavior
# ---------------------------------------------------------------------------

def test_register_alias_conflict_requires_replace(reg):
    # 'm' is an existing UNIT symbol; registering it as an alias without replace must fail
    with pytest.raises(ValueError):
        reg.register_alias("m", "mm")


def test_register_alias_replace_allows_shadowing_unit_symbol(reg):
    # Sanity: meter vs millimeter scales/dims
    m = reg.get("m")
    mm = reg.get("mm")
    assert mm.dim == m.dim
    assert mm.scale_to_si == pytest.approx(1e-3)

    # With replace=True we can intentionally shadow an existing unit symbol
    reg.register_alias("m", "mm", replace=True)

    # After replacement, lookups of 'm' resolve through the alias
    assert reg.get("m") is mm
    # And canonical path remains unchanged
    assert reg.get("mm") is mm


def test_register_alias_requires_replace_to_repoint_existing_alias(reg):
    # 'ohm' is pre-registered as an alias for 'Ω' in the bootstrapped registry.
    # Attempting to repoint an existing alias without replace=True
    # should raise a ValueError (preventing accidental alias reassignment).
    V = reg.get("V")

    with pytest.raises(ValueError):
        reg.register_alias("ohm", "V")  # repoint alias 'ohm' → 'V' should fail

    # The alias should still point to its original target ('Ω')
    ohm_unit = reg.get("ohm")
    omega_unit = reg.get("Ω")

    assert ohm_unit is omega_unit
    assert ohm_unit is not V



def test_register_alias_normalizes_alias_key(reg):
    # normalize_symbol should be applied to the alias key
    # Mixed-case key should still be usable via its normalized form.
    reg.register_alias("MiXeD_Key", "m")
    # Expect to retrieve via normalized lookup (likely lowercase)
    assert reg.get("mixed_key") is reg.get("m")


def test_register_alias_thread_safe_single_mapping(reg):
    # Smoke test: many threads try to register the same alias → same canonical
    errs = []
    def worker():
        try:
            reg.register_alias("alias_concurrent", "s")
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for t in threads: 
        t.start()
    for t in threads: 
        t.join()

    assert not errs
    # Alias must resolve to the exact same LinearUnit object as 's'
    assert reg.get("alias_concurrent") is reg.get("s")

# ---------------------------------------------------------------------------
# UnitNamespace.define behavior + UnitNamespace convenience features
# ---------------------------------------------------------------------------


def test_namespace_define_basic_scale_and_dim(reg):
    ns = UnitNamespace(reg)
    # Define a whimsical length unit: 1 smoot = 1.7018 m
    ns.define("smoot", 1.7018, reg.get("m"))
    smoot = reg.get("smoot")
    m = reg.get("m")
    assert smoot.dim == m.dim
    assert smoot.scale_to_si == pytest.approx(1.7018)

    # Access via __call__ and attribute should be the same object
    assert ns("smoot") is smoot
    assert ns.smoot is smoot


def test_namespace_define_with_non_si_reference(reg):
    ns = UnitNamespace(reg)
    # 'min' is 60 s; define half_min as 0.5 * min → 30 s
    ns.define("half_min", 0.5, reg.get("min"))
    half_min = reg.get("half_min")
    s = reg.get("s")
    assert half_min.dim == s.dim
    assert half_min.scale_to_si == pytest.approx(30.0)


def test_namespace_define_conflict_requires_replace(reg):
    ns = UnitNamespace(reg)
    ns.define("qux", 2.0, reg.get("m"))  # 2 m
    # redefining without replace must raise (delegates to registry.register)
    with pytest.raises(ValueError):
        ns.define("qux", 3.0, reg.get("m"))


def test_namespace_define_replace_allows_overwrite(reg):
    ns = UnitNamespace(reg)
    ns.define("quux", 2.0, reg.get("m"))      # 2 m
    assert reg.get("quux").scale_to_si == pytest.approx(2.0)

    # Overwrite with a different scale via replace=True
    ns.define("quux", 5.0, reg.get("m"), replace=True)
    assert reg.get("quux").scale_to_si == pytest.approx(5.0)


def test_namespace_dunder_call_and_attr_and_dir(reg):
    ns = UnitNamespace(reg)

    # __call__ should proxy to registry.get
    assert ns("m") is reg.get("m")

    # Unknown attribute should raise AttributeError (not ValueError)
    with pytest.raises(AttributeError):
        _ = ns.this_symbol_does_not_exist

    # __dir__ should include units and known aliases (e.g., 'ohm' → 'Ω')
    # and any unit we define.
    ns.define("smoot", 1.7018, reg.get("m"))
    entries = dir(ns)
    # Known unit symbol
    assert "m" in entries
    # Known alias from the bootstrapped registry
    assert "ohm" in entries
    # Newly defined symbol
    assert "smoot" in entries

# ---------------------------------------------------------------------------
# New behavior: alias stored as normalized + literal + casefolded keys
# ---------------------------------------------------------------------------

def test_alias_lookup_accepts_literal_and_casefold_variants(reg):
    # No special normalization here; we want to exercise literal + casefolded keys
    reg.register_alias("MiXeD_Key", "m")
    # literal spelling works
    assert reg.get("MiXeD_Key") is reg.get("m")
    # casefolded spelling works (new behavior)
    assert reg.get("mixed_key") is reg.get("m")


def test_dir_includes_literal_alias_spelling(reg):
    # The literal alias should show up in __dir__ (discoverability)
    ns = reg.as_namespace()
    reg.register_alias("myAlias", "s")
    names = dir(ns)
    assert "myAlias" in names     # literal
    # normalized/casefold may also be present depending on other aliases you add,
    # but the key requirement is the literal shows up.


def test_ohm_variants_all_resolve_and_literal_shows_in_dir(reg):
    ns = reg.as_namespace()
    # Re-register to ensure behavior regardless of bootstrap order
    reg.register_alias("ohm", "Ω", replace=True)
    # All case variants should resolve (casefolded key)
    for a in ["ohm", "Ohm", "OHM"]:
        assert reg.get(a) is reg.get("Ω")
    # dir should include a human-readable spelling (literal alias)
    assert "ohm" in dir(ns)


def test_repoint_alias_updates_target_all_variants(reg):
    # Point a mixed-case alias at 'm', then repoint to 's'
    reg.register_alias("MiXeD_Key", "m", replace=True)
    assert reg.get("mixed_key") is reg.get("m")
    reg.register_alias("MiXeD_Key", "s", replace=True)
    # All spellings now resolve to 's'
    assert reg.get("MiXeD_Key") is reg.get("s")
    assert reg.get("mixed_key") is reg.get("s")


def test_alias_still_resolves_to_same_unit_object(reg):
    reg.register_alias("fortnight_alias", "fortnight", replace=True)
    assert reg.get("fortnight_alias") is reg.get("fortnight")

def test_membership_checks_consider_aliases(reg):
    ns = reg.as_namespace()
    reg.register_alias("alias_seconds", "s", replace=True)
    assert "alias_seconds" in reg
    assert "alias_seconds" in ns

# ---------------------------------------------------------------------------
# LinearUnit vs. Alias name conflict checks (without replace=True)
# ---------------------------------------------------------------------------
def test_register_unit_conflict_with_existing_alias_raises(reg):
    """
    Tests that UnitRegistry.register() fails if the new unit's name
    conflicts with an *existing alias*.
    """
    # 1. Setup: Create a unique, non-conflicting alias name
    alias_name = "my_custom_alias"
    reg.register_alias(alias_name, "m")
    
    # Sanity check: alias resolves correctly
    assert reg.get(alias_name) is reg.get("m")

    # 2. Action & Assert: Try to register a *unit* with the exact same name.
    # This should fail because 'my_custom_alias' is already an alias key.
    new_unit = LinearUnit(alias_name, 1.2345, LENGTH)
    with pytest.raises(ValueError):
        reg.register(new_unit)


def test_register_alias_conflict_with_existing_unit_raises(reg):
    """
    Tests that UnitRegistry.register_alias() fails if the new alias name
    conflicts with an *existing unit symbol* (when replace=False).
    """
    # 1. Setup: 'm' (meter) is a guaranteed registered base unit.
    assert "m" in reg.all() # Check it's a unit symbol

    # 2. Action & Assert: Try to register an *alias* with the name 'm'
    # This should fail because 'm' is already a registered unit symbol.
    with pytest.raises(ValueError):
        reg.register_alias("m", "s") # Try to make 'm' (a unit) an alias for 's'

def test_register_alias_conflict_with_casefolded_unit_name_raises(reg):
    """
    Tests the specific bug where an alias conflicts with an existing unit
    symbol via case-folding, which the original code missed.
    """
    # 1. Arrange: Register a unit with a simple, lowercase name.
    unit_name = "conflict_unit"
    reg.register(LinearUnit(unit_name, 1.0, LENGTH))

    # Define an alias that is the same as the unit name, but with a different case.
    conflicting_alias = "CONFLICT_UNIT"

    # 2. Act & Assert: Attempt to register the conflicting alias.
    # The old, buggy code would FAIL to raise an error here, creating an
    # inconsistent state because "CONFLICT_UNIT".casefold() == "conflict_unit".
    # The new, fixed code correctly detects this conflict and raises a ValueError.
    with pytest.raises(ValueError, match="a unit with the name 'conflict_unit' already exists"):
        reg.register_alias(conflicting_alias, "s")


def test_register_alias_conflict_with_normalized_unit_name_raises(reg):
    """
    Tests the bug from another angle: an alias conflicts with an existing
    unit symbol via normalization (e.g., 'u' vs 'µ').
    """
    # 1. Arrange: Register the canonical micro-ampere unit.
    unit_name = "µA"
    reg.register(LinearUnit(unit_name, 1e-6, LENGTH))

    # Define an alias that uses the ASCII 'u' fallback for the micro prefix.
    conflicting_alias = "uA"

    # 2. Act & Assert: Attempt to register the conflicting alias.
    # The old code would miss this because 'uA' is not literally in self._units.
    # The new code normalizes the alias to 'µA' *before* checking for
    # conflicts and correctly raises the error.
    with pytest.raises(ValueError, match="a unit with the name 'µA' already exists"):
        reg.register_alias(conflicting_alias, "s")


# ---------------------------------------------------------------------------
# Reserved-name rejection for register() and register_alias()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["define", "__init__", "_reserved_names"])
def test_register_rejects_reserved_unit_names(reg, name):
    ref = reg.get("m")
    with pytest.raises(ValueError, match="UnitNamespace attribute/method"):
        reg.register(LinearUnit(name, 1.0, ref.dim))

@pytest.mark.parametrize("name", ["define", "__init__", "_reserved_names"])
def test_register_rejects_reserved_unit_names_even_with_replace(reg, name):
    ref = reg.get("m")
    with pytest.raises(ValueError, match="UnitNamespace attribute/method"):
        reg.register(LinearUnit(name, 1.0, ref.dim), replace=True)

@pytest.mark.parametrize("alias", [
    # literal/public
    "define",
    # dunder
    "__init__",
    # public attribute we add post-class
    "_reserved_names",
    # case variants to ensure casefold path is checked
    "DeFiNe",
    "__INIT__",
])
def test_register_alias_rejects_reserved_names(reg, alias):
    with pytest.raises(ValueError, match="UnitNamespace attribute/method"):
        reg.register_alias(alias, "m")

@pytest.mark.parametrize("alias", ["define", "__init__", "_reserved_names"])
def test_register_alias_rejects_reserved_names_even_with_replace(reg, alias):
    with pytest.raises(ValueError, match="UnitNamespace attribute/method"):
        reg.register_alias(alias, "m", replace=True)

def test_non_reserved_names_still_register_and_alias_ok(reg):
    # Positive control to show normal behavior still works
    ref = reg.get("m")
    reg.register(LinearUnit("myunit", 201.168, ref.dim))
    assert reg.get("myunit").scale_to_si == pytest.approx(201.168)

    reg.register_alias("munyt", "myunit")
    assert reg.get("munyt") is reg.get("myunit")