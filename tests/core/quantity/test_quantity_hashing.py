from __future__ import annotations
from quantium.core.dimensions import LENGTH, MASS
from quantium.core.quantity import LinearQuantity
from quantium.core.unit import LinearUnit
from quantium.units.registry import DEFAULT_REGISTRY as ureg
from quantium.units import u
import pytest


def test_as_key_basic():
    """Tests the basic functionality of as_key."""
    q1 = LinearQuantity(100.0, u.cm) # _mag_si = 1.0
    key = q1.as_key()
    assert key == (LENGTH, 1.0)
    
    q2 = LinearQuantity(5.5, u.g) # _mag_si = 0.0055
    key2 = q2.as_key()
    assert key2 == (MASS, 0.0055)

def test_as_key_default_precision_grouping():
    """
    Tests that quantities "fuzzy equal" at the default precision (12)
    produce the same key.
    """
    # These two are "equal" according to __eq__
    q1 = LinearQuantity(1.0, u.m)
    q2 = LinearQuantity(1.0 + 1e-13, u.m) # 1.0000000000001
    q3 = LinearQuantity(1.0 - 1e-13, u.m) # 0.9999999999999

    # At precision=12, all should round to 1.0
    key1 = q1.as_key()
    key2 = q2.as_key()
    key3 = q3.as_key()
    
    assert key1 == (LENGTH, 1.0)
    assert key1 == key2
    assert key1 == key3

def test_as_key_default_precision_separation():
    """
    Tests that quantities that are *not* "fuzzy equal"
    produce different keys at the default precision.
    """
    q1 = LinearQuantity(1.0, u.m)
    q2 = LinearQuantity(1.0 + 1e-9, u.m)  # 1.000000001
    
    key1 = q1.as_key() # (DIM_LENGTH, 1.0)
    key2 = q2.as_key() # (DIM_LENGTH, 1.000000001)
    
    assert key1 != key2

def test_as_key_custom_precision():
    """
    Tests that the `precision` argument correctly changes the rounding.
    """
    q1 = LinearQuantity(1.2345678, u.m)
    q2 = LinearQuantity(1.2345679, u.m)
    
    # At default precision (12), they are different
    key1_default = q1.as_key()
    key2_default = q2.as_key()
    assert key1_default != key2_default

    # At a lower precision (6), they are equal
    key1_p6 = q1.as_key(precision=6) # (DIM_LENGTH, 1.234568)
    key2_p6 = q2.as_key(precision=6) # (DIM_LENGTH, 1.234568)
    assert key1_p6 == key2_p6 # These ARE equal

    # Test grouping at precision 3
    q3 = LinearQuantity(1.2341, u.m) # This was 1.2345
    q4 = LinearQuantity(1.2342, u.m) # This was 1.2346

    key3_p3 = q3.as_key(precision=3) # (DIM_LENGTH, 1.234)
    key4_p3 = q4.as_key(precision=3) # (DIM_LENGTH, 1.234)
    assert key3_p3 == key4_p3 # They group at precision 3

    # Test separation at precision 4
    key3_p4 = q3.as_key(precision=4) # (DIM_LENGTH, 1.2341)
    key4_p4 = q4.as_key(precision=4) # (DIM_LENGTH, 1.2342)
    assert key3_p4 != key4_p4 # They separate at precision 4

def test_as_key_zero_handling():
    """Tests that -0.0 and 0.0 hash to the same key."""
    q_pos_zero = LinearQuantity(0.0, u.m)
    q_neg_zero = LinearQuantity(-0.0, u.m)
    
    # Check their internal values to confirm one is -0.0
    # Note: This behavior depends on the python env, but generally holds
    # assert str(q_neg_zero._mag_si) == "-0.0"
    
    key_pos = q_pos_zero.as_key()
    key_neg = q_neg_zero.as_key()
    
    assert key_pos == (LENGTH, 0.0)
    assert key_neg == (LENGTH, 0.0)
    assert key_pos == key_neg
    
    # Check the hash directly
    assert hash(key_pos) == hash(key_neg)

def test_as_key_in_dictionary():
    """A practical test of using as_key() to group items in a dict."""
    
    counts = {}
    
    # 1.0 m
    q_m = LinearQuantity(1.0, u.m)
    
    # 1.0 m (with FP noise)
    q_cm_noise = LinearQuantity(100.0000000000001, u.cm)
    
    # 1.001 m
    q_mm = LinearQuantity(1001.0, u.mm)
    
    # --- Use default precision (12) ---
    key_m = q_m.as_key()
    key_cm_noise = q_cm_noise.as_key()
    key_mm = q_mm.as_key()

    counts[key_m] = 1
    
    # q_cm_noise should find the same key as q_m
    if key_cm_noise in counts:
        counts[key_cm_noise] += 1
    else:
        counts[key_cm_noise] = 1
        
    # q_mm should be a new key
    if key_mm in counts:
        counts[key_mm] += 1
    else:
        counts[key_mm] = 1

    # We should have two entries: one for 1.0, one for 1.001
    assert len(counts) == 2
    assert counts[key_m] == 2
    assert counts[key_mm] == 1
    assert counts[key_cm_noise] == 2 # Same as key_m
