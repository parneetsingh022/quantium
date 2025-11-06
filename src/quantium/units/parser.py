"""
quantium.units.parser
"""

from functools import lru_cache
from fractions import Fraction
from typing import Tuple, Union, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quantium.core.unit import LinearUnit  # <-- Import LinearUnit for type hints
    from quantium.units.registry import UnitsRegistry

# --- Plan node types ------------------------------------------------
# ("name", <str>)
# ("pow", <plan>, <int|Fraction>)
# ("mul", <plan>, <plan>)
# ("div", <plan>, <plan>)
Plan = Tuple[str, Union[str, "Plan"], Union[int, Fraction, "Plan", None]]

# ---------------- Parser that builds a PLAN (no registry lookups!) ----------------
class _UnitExprParser:
    """Recursive-descent parser for unit expressions with integer and rational exponents."""
    def __init__(self, text: str):
        self.s = text
        self.n = len(text)
        self.i = 0

    def parse(self) -> Plan:
        plan = self._parse_expr()
        self._skip_ws()
        if self.i != self.n:
            raise ValueError(
                f"Unexpected trailing input at {self.i}: {self.s[self.i:self.i+10]!r}"
            )
        return plan

    # expr := term (('*' | '/') term)*
    def _parse_expr(self) -> Plan:
        left = self._parse_term()
        while True:
            self._skip_ws()
            if self._peek('*') and not self._peek('**'):
                self._eat('*')
                right = self._parse_term()
                left = ("mul", left, right)
            elif self._peek('/'):
                self._eat('/')
                right = self._parse_term()
                left = ("div", left, right)
            else:
                break
        return left

    # term := factor ( ('**' exponent) | ('^' exponent) )*
    def _parse_term(self) -> Plan:
        base = self._parse_factor()
        while True:
            self._skip_ws()
            op = '**' if self._peek('**') else ('^' if self._peek('^') else None)
            if not op:
                break
            self._eat(op)
            #exp = Fraction(self._parse_signed_int(), 1) if op == '**' else self._parse_caret_exponent()
            exp = self._parse_caret_exponent()
            base = ("pow", base, exp)
        return base

    # factor := NAME | '(' expr ')'
    def _parse_factor(self) -> Plan:
        self._skip_ws()
        if self._peek('('):
            self._eat('(')
            val = self._parse_expr()
            self._skip_ws()
            self._eat(')')
            return val

        # Allow a literal "1" to represent a dimensionless unit
        if self.i < self.n and self.s[self.i] == '1':
            self.i += 1
            return ("one", "1", None)

        name = self._parse_name()
        if not name:
            ch = self.s[self.i:self.i+1]
            raise ValueError(f"Expected unit name or '(' at {self.i}, got {ch!r}")
        return ("name", name, None)

    # ---- token helpers ----
    def _parse_name(self) -> Optional[str]:
        self._skip_ws()
        i0 = self.i
        if i0 >= self.n:
            return None
        ch0 = self.s[i0]
        # allow Unicode letter or underscore as the first character
        if not (ch0.isalpha() or ch0 == '_'):
            return None

        self.i += 1
        # characters allowed after the first one
        EXTRA_NAME_CHARS = {'°', 'µ', 'Ω', 'Δ'}
        while self.i < self.n:
            ch = self.s[self.i]
            if ch.isalnum() or ch == '_' or ch in EXTRA_NAME_CHARS:
                self.i += 1
            else:
                break
        return self.s[i0:self.i]

    def _parse_signed_int(self) -> int:
        self._skip_ws()
        i0 = self.i
        if self.i < self.n and self.s[self.i] in '+-':
            self.i += 1
        i1 = self.i
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
        if i1 == self.i:
            raise ValueError(f"Expected integer exponent at {self.i}")
        return int(self.s[i0:self.i])

    def _parse_unsigned_int(self) -> int:
        self._skip_ws()
        i0 = self.i
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
        if i0 == self.i:
            raise ValueError(f"Expected unsigned integer at {self.i}")
        return int(self.s[i0:self.i])

    def _parse_caret_exponent(self) -> Fraction:
        self._skip_ws()
        if self._peek('('):
            self._eat('(')
            numer = self._parse_signed_int()
            self._skip_ws()
            if self._peek(')'):
                self._eat(')')
                return Fraction(numer, 1)
            self._eat('/')
            denom = self._parse_unsigned_int()
            if denom == 0:
                raise ValueError("Denominator of fractional exponent cannot be zero")
            self._skip_ws()
            self._eat(')')
            return Fraction(numer, denom)

        sign = 1
        if self._peek('-'):
            self._eat('-')
            sign = -1
        elif self._peek('+'):
            self._eat('+')
        self._skip_ws()
        i0 = self.i
        while self.i < self.n and self.s[self.i].isdigit():
            self.i += 1
        if i0 == self.i:
            raise ValueError(f"Expected numeric exponent after '^' at {self.i}")
        value = int(self.s[i0:self.i])
        return Fraction(sign * value, 1)

    def _skip_ws(self) -> None:  # <-- FIX: Added return type
        s, n, i = self.s, self.n, self.i
        while i < n and s[i].isspace():
            i += 1
        self.i = i

    def _peek(self, tok: str) -> bool:
        self._skip_ws()
        if tok == '**':
            return self.s[self.i:self.i+2] == '**'
        return self.i < self.n and self.s[self.i] == tok

    def _eat(self, tok: str) -> None:  # <-- FIX: Added return type
        if not self._peek(tok):
            got = self.s[self.i:self.i+len(tok)]
            raise ValueError(f"Expected {tok!r} at {self.i}, got {got!r}")
        self.i += len(tok)

# ---------------- Evaluation of a plan against a given registry ----------------
def _eval_plan(plan: Plan, reg: "UnitsRegistry") -> "LinearUnit":  # <-- FIX: Added return type
    kind, op1, op2 = plan

    if kind == "name":
        if not isinstance(op1, str):
            raise ValueError(f"Malformed 'name' plan (expected str): {plan!r}")
        try:
            u = reg.get(op1)
            # Only linear units are valid in algebraic expressions
            from quantium.core.unit import LinearUnit
            if not isinstance(u, LinearUnit):
                raise ValueError(f"Unit '{op1}' is not a linear unit and cannot be used in expressions")
            return u
        except Exception as e:
            raise ValueError(f"Unknown unit '{op1}': {e}") from None

    elif kind == "one":
        from quantium.core.unit import LinearUnit
        from quantium.core.dimensions import DIM_0
        return LinearUnit("1", 1.0, DIM_0)

    elif kind == "pow":
        if not isinstance(op1, tuple):
            raise ValueError(f"Malformed 'pow' plan (expected plan tuple): {plan!r}")
        if not isinstance(op2, (int, Fraction)):
            raise ValueError(f"Malformed 'pow' plan (expected int or Fraction exponent): {plan!r}")
        base = _eval_plan(op1, reg)
        return base ** op2

    elif kind == "mul":
        if not isinstance(op1, tuple):
            raise ValueError(f"Malformed 'mul' plan (left must be plan tuple): {plan!r}")
        if not isinstance(op2, tuple):
            raise ValueError(f"Malformed 'mul' plan (right must be plan tuple): {plan!r}")
        left = _eval_plan(op1, reg)
        right = _eval_plan(op2, reg)
        return left * right

    elif kind == "div":
        if not isinstance(op1, tuple):
            raise ValueError(f"Malformed 'div' plan (left must be plan tuple): {plan!r}")
        if not isinstance(op2, tuple):
            raise ValueError(f"Malformed 'div' plan (right must be plan tuple): {plan!r}")
        left = _eval_plan(op1, reg)
        right = _eval_plan(op2, reg)
        return left / right

    else:
        raise RuntimeError(f"Invalid plan node: {plan!r}")

# ---------------- Public API with caching-safe compilation ----------------
@lru_cache(maxsize=4096)
def _compile_unit_expr(expr: str) -> Plan:
    disallowed = set('~!@#$%&|+=,:;?<>\'"`\\[]{}')
    if any(c in disallowed for c in expr):
        raise ValueError(
            "Only *, /, **, ^, parentheses, unit names, and integer or rational exponents are allowed."
        )
    return _UnitExprParser(expr).parse()

def extract_unit_expr(expr: str, reg: "UnitsRegistry") -> "LinearUnit":  # <-- FIX: Added return type
    """
    Fast custom parser for unit expressions like 'kg*m/(nF**2 * s**2)'.
    """
    plan = _compile_unit_expr(expr)
    return _eval_plan(plan, reg)
