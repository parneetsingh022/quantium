# pytest tests for quantium.units.namespace (UnitNamespace)

import pytest

import quantium.units.registry as regmod
from quantium.units.registry import UnitsRegistry, UnitNamespace
from quantium.core.unit import LinearUnit  # your LinearUnit class
from quantium.core.dimensions import dim_div
from quantium.units.registry import _bootstrap_default_registry

@pytest.fixture
def u():
    # fresh namespace per test to avoid global state bleed
    return _bootstrap_default_registry().as_namespace()


def test_import_u_from_quantium(monkeypatch, reg):
    """Ensure `from quantium.units import u` works and is bound to DEFAULT_REGISTRY."""
    import importlib
    import quantium.units.registry as regmod

    # Patch DEFAULT_REGISTRY before reload so `quantium.u` binds to it
    monkeypatch.setattr(regmod, "DEFAULT_REGISTRY", reg, raising=True)

    import quantium
    importlib.reload(quantium)  # rebinds u to the patched registry

    # Verify `u` is importable directly
    from quantium.units import u
    assert hasattr(u, "_reg")
    assert u._reg is reg  # should wrap the patched registry

    # Sanity check: using u returns real LinearUnit objects
    m = u.m
    s = u.s
    assert m.name == "m"
    assert s.name == "s"
    assert (u("m/s").dim == dim_div(m.dim , s.dim))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def reg():
    """Fresh, fully-bootstrapped UnitsRegistry for isolation per test."""
    return regmod._bootstrap_default_registry()

@pytest.fixture()
def ns(reg):
    """A UnitNamespace over an isolated registry."""
    return UnitNamespace(reg)

# ---------------------------------------------------------------------------
# Access styles: __call__, __getitem__, __getattr__
# ---------------------------------------------------------------------------

def test_namespace_call_returns_unit(ns, reg):
    u1 = ns("m")
    u2 = reg.get("m")
    assert isinstance(u1, LinearUnit)
    assert u1 is u2

def test_namespace_getattr_returns_unit(ns, reg):
    u1 = ns.kg
    u2 = reg.get("kg")
    assert isinstance(u1, LinearUnit)
    assert u1 is u2

def test_namespace_access_styles_equivalent(ns):
    assert ns("A") is ns.A

# ---------------------------------------------------------------------------
# Aliases and normalization also work via __getattr__
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias, canonical", [
    ("ohm", "Ω"),
    ("Ohm", "Ω"),
    ("OHM", "Ω"),
])
def test_namespace_getattr_aliases(ns, reg, alias, canonical):
    assert ns.__getattr__(alias) is reg.get(canonical)

# ---------------------------------------------------------------------------
# Compound expressions route through registry.get
# ---------------------------------------------------------------------------

def test_namespace_call_compound_expression(ns, reg):
    a = ns("m/s**2")         # via __call__
    b = reg.get("m") / (reg.get("s") ** 2)
    assert a.dim == b.dim
    assert a.scale_to_si == pytest.approx(b.scale_to_si)

# ---------------------------------------------------------------------------
# Error behavior
# ---------------------------------------------------------------------------

def test_namespace_getattr_unknown_raises_attributeerror(ns):
    with pytest.raises(AttributeError):
        _ = ns.__getattr__("blorp")

def test_namespace_call_unknown_raises_valueerror(ns):
    with pytest.raises(ValueError):
        _ = ns("blorp")

# ---------------------------------------------------------------------------
# __dir__: should include units and aliases (best-effort)
# ---------------------------------------------------------------------------

def test_namespace_dir_includes_registered_symbols(ns, reg):
    # register a custom unit and an alias to see them in dir()
    reg.register(LinearUnit("ft", 0.3048, reg.get("m").dim))
    reg.register_alias("foot", "ft")

    names = dir(ns)
    # Core units should be present
    assert "m" in names
    assert "s" in names
    # Newly registered + alias should appear
    assert "ft" in names
    assert "foot" in names

def test_namespace_dir_is_sorted_and_not_empty(ns):
    names = dir(ns)
    assert names == sorted(names)
    assert len(names) > 10  # sanity: we expect lots of symbols

# ---------------------------------------------------------------------------
# Top-level convenience 'u' wiring smoke test
# ---------------------------------------------------------------------------

def test_top_level_u_uses_default_registry(monkeypatch, reg):
    # Patch DEFAULT_REGISTRY before importing the package root
    monkeypatch.setattr(regmod, "DEFAULT_REGISTRY", reg, raising=True)

    # Re-import package root to bind u to the patched registry
    import importlib, quantium
    importlib.reload(quantium)

    from quantium.units import u
    assert u.m is reg.get("m")
    assert u("cm").dim == reg.get("cm").dim

# ---------------------------------------------------------------------------
# Validation to ensure the `define` method prevents registration of units
# whose names conflict with existing UnitNamespace attributes or methods.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["define", "__init__", "_reserved_names"])
def test_define_raises_value_error_when_conflicted_with_namespace_attributes(u, name):
    # any valid reference unit is fine
    with pytest.raises(ValueError, match="conflicts with UnitNamespace"):
        u.define(name, 1.0, u.m)

def test_define_nonconflicting_name_succeeds(u):
    u.define("test", 201.168, u.m)
    assert u("test").scale_to_si == pytest.approx(201.168)
    # attribute access should also resolve if not colliding
    assert u.test is u("test")

def test_registry_blocks_reserved_name_on_register():
    reg = _bootstrap_default_registry()
    u = reg.as_namespace()
    # Try to register directly via the registry (bypassing UnitNamespace.define)
    from quantium.core.unit import LinearUnit
    with pytest.raises(ValueError, match="UnitNamespace"):
        reg.register(LinearUnit("define", 1.0, u.m.dim))

def test_registry_blocks_reserved_name_on_alias():
    reg = _bootstrap_default_registry()
    # Aliases that collide with UnitNamespace should also fail
    with pytest.raises(ValueError, match="UnitNamespace"):
        reg.register_alias("__init__", "m")
