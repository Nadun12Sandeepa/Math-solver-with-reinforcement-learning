import sympy as sp
import numpy as np
import re
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

ACTIONS = [
    "expand",          # 0
    "simplify",        # 1
    "move_constant",   # 2
    "move_variable",   # 3
    "solve"            # 4
]

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

class MathEnv:
    def __init__(self, equation):
        self.x = sp.symbols("x")
        self.original_eq = equation
        self.eq = self._parse_equation(equation)
        self.steps = 0
        self.max_steps = 6

    # -------------------------------
    # Robust equation parser
    # -------------------------------
    def _clean(self, s):
        # remove unicode spaces
        s = s.replace("\u202f", " ").replace("\u2009", " ").strip()
        # normalize minus signs
        s = s.replace("−", "-")
        return s

    def _parse_side(self, side):
        side = self._clean(side)
        return parse_expr(side, transformations=TRANSFORMS)

    def _parse_equation(self, eq_str):
        if "=" not in eq_str:
            raise ValueError("Equation must contain '='")

        left, right = eq_str.split("=")
        left_expr = self._parse_side(left)
        right_expr = self._parse_side(right)
        return sp.Eq(left_expr, right_expr)

    # -------------------------------
    # RL state encoding
    # -------------------------------
    def encode_state(self):
        text = str(self.eq)
        vec = np.zeros(128)
        for i, c in enumerate(text[:128]):
            vec[i] = ord(c) / 128
        return vec

    # -------------------------------
    # Equivalence check
    # -------------------------------
    def _equivalent(self, eq1, eq2):
        return sp.simplify(eq1.lhs - eq1.rhs) == sp.simplify(eq2.lhs - eq2.rhs)

    # -------------------------------
    # Environment step
    # -------------------------------
    def step(self, action):
        self.steps += 1
        reward = -1
        done = False
        old_eq = self.eq

        try:
            if action == 0:  # expand
                new_eq = sp.Eq(sp.expand(old_eq.lhs), sp.expand(old_eq.rhs))

            elif action == 1:  # simplify
                new_eq = sp.Eq(sp.simplify(old_eq.lhs), sp.simplify(old_eq.rhs))

            elif action == 2:  # move constant
                const = old_eq.lhs.as_independent(self.x)[0]
                new_eq = sp.Eq(old_eq.lhs - const, old_eq.rhs - const)

            elif action == 3:  # move variable
                var = old_eq.lhs.as_independent(self.x)[1]
                new_eq = sp.Eq(old_eq.lhs - var, old_eq.rhs - var)

            elif action == 4:  # solve
                sol = sp.solve(old_eq, self.x)
                if sol:
                    reward = 20 - self.steps
                    done = True
                return self.encode_state(), reward, done

            else:
                new_eq = old_eq

            if self._equivalent(old_eq, new_eq):
                self.eq = new_eq
                reward = 3
            else:
                reward = -5

        except Exception:
            reward = -10

        if self.steps >= self.max_steps:
            done = True

        return self.encode_state(), reward, done
