from dataclasses import FrozenInstanceError
import math

from quantium.core.dimensions import LENGTH, TEMPERATURE, TIME, dim_div
from quantium.core.quantity import LinearQuantity
from quantium.core.unit import LinearUnit
from quantium.units.registry import DEFAULT_REGISTRY as ureg

# -------------------------------
# to_si(): preferred symbol & fallback
# -------------------------------

def test_to_si_uses_preferred_symbol_when_available(monkeypatch):
    # Arrange: make preferred_symbol_for_dim return a symbol for LENGTH
    import quantium.core.utils as utils
    # Monkeypatch utils functions that to_si imports locally
    def fake_preferred(dim):
        return "m" if dim == LENGTH else None
    def fake_format(dim):
        return "LENGTH?"  # should not be used in this test

    monkeypatch.setattr(utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(utils, "format_dim", fake_format, raising=True)

    cm = LinearUnit("cm", 0.01, LENGTH)
    q_si = (123 * cm).to_si()

    assert isinstance(q_si, LinearQuantity)
    assert q_si.unit.name == "m"         # preferred symbol chosen
    assert q_si.unit.scale_to_si == 1.0  # SI unit
    # magnitudes in SI should match _mag_si:
    assert math.isclose(q_si._mag_si, 1.23)

def test_to_si_fallbacks_to_formatted_dim_when_no_symbol(monkeypatch):
    import quantium.core.utils as utils
    def fake_preferred(dim):
        return None  # force fallback
    def fake_format(dim):
        # For LENGTH/TEMPERATURE, produce a composed name:
        return "m/s" if dim == dim_div(LENGTH, TEMPERATURE) else "1"

    monkeypatch.setattr(utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(utils, "format_dim", fake_format, raising=True)

    m = LinearUnit("m", 1.0, LENGTH)
    s = LinearUnit("s", 1.0, TEMPERATURE)
    q = ((5 * m) / (2 * s)).to_si()

    assert q.unit.name == "m/s"          # composed SI name from format_dim
    assert q.unit.scale_to_si == 1.0
    assert math.isclose(q._mag_si, 2.5)

# -------------------------------
# .si property
# -------------------------------

def test_si_equivalent_to_to_si():
    cm = ureg.get("cm")

    q = 123 * cm        # 1.23 m in SI
    q_si = q.si
    q_to_si = q.to_si()

    # Same dim and SI scale
    assert q_si.unit.scale_to_si == 1.0
    assert q_si.dim == q_to_si.dim
    assert q_si.unit == q_to_si.unit
    assert math.isclose(q_si._mag_si, q_to_si._mag_si, rel_tol=1e-12, abs_tol=0.0)


def test_si_uses_preferred_symbol_when_available(monkeypatch):
    # Arrange: make preferred_symbol_for_dim return a symbol for LENGTH
    import quantium.core.utils as utils

    def fake_preferred(dim):
        return "m" if dim == LENGTH else None

    def fake_format(dim):
        return "LENGTH?"  # should not be used in this test

    monkeypatch.setattr(utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(utils, "format_dim", fake_format, raising=True)

    cm = ureg.get("cm")
    q_si = (123 * cm).si  # 1.23 m

    assert isinstance(q_si, LinearQuantity)
    assert q_si.unit.name == "m"         # preferred symbol chosen
    assert q_si.unit.scale_to_si == 1.0  # SI unit
    assert math.isclose(q_si._mag_si, 1.23)


def test_si_fallbacks_to_formatted_dim_when_no_symbol(monkeypatch):
    import quantium.core.utils as utils

    def fake_preferred(dim):
        return None  # force fallback

    def fake_format(dim):
        # composed SI name for velocity
        return "m/s" if dim == dim_div(LENGTH, TIME) else "1"

    monkeypatch.setattr(utils, "preferred_symbol_for_dim", fake_preferred, raising=True)
    monkeypatch.setattr(utils, "format_dim", fake_format, raising=True)

    m = ureg.get("m")
    s = ureg.get("s")
    q = ((5 * m) / (2 * s)).si  # 2.5 m/s

    assert q.unit.name == "m/s"
    assert q.unit.scale_to_si == 1.0
    assert math.isclose(q._mag_si, 2.5)


def test_si_does_not_mutate_original_quantity():
    cm = ureg.get("cm")
    ms = ureg.get("ms")

    v = 1000 * (cm / ms)  # original in cm/ms
    v_si = v.si           # 10000 m/s

    # original unchanged
    assert v.unit == (cm / ms)
    # new object is SI
    assert v_si.unit.scale_to_si == 1.0
    assert math.isclose(v_si._mag_si, 10000.0)


def test_si_repr_respects_preferred_symbol_when_scale_is_1(monkeypatch):
    # Ensure repr of q.si upgrades to the preferred symbol (since SI scale is 1.0)
    import quantium.core.utils as utils

    # prettifier passthrough to assert exact string
    monkeypatch.setattr(utils, "prettify_unit_name_supers", lambda s, cancel=True: s, raising=True)
    # preferred symbol for LENGTH is "m"
    monkeypatch.setattr(utils, "preferred_symbol_for_dim", lambda d: "m" if d == LENGTH else None, raising=True)

    cm = ureg.get("cm")
    q_si = (123 * cm).si  # 1.23 m in SI
    assert repr(q_si) == "1.23 m"
