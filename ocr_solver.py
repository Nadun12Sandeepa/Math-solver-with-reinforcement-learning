import sympy as sp
import pytesseract
import re
from PIL import Image
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# 🔧 SET TESSERACT PATH (CHANGE IF NEEDED)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

x = sp.symbols("x")

def clean_text(text):
    text = text.replace("−", "-")
    text = re.sub(r"[^\w\s\+\-\=\(\)\*\/\.]", "", text)
    return text.strip()

def parse_equation(eq):
    left, right = eq.split("=")
    l = parse_expr(left, transformations=TRANSFORMS)
    r = parse_expr(right, transformations=TRANSFORMS)
    return sp.Eq(l, r)

def solve_equation(eq_text):
    eq_text = clean_text(eq_text)
    eq = parse_equation(eq_text)

    steps = []
    steps.append(f"Original: {eq}")

    expanded = sp.Eq(sp.expand(eq.lhs), sp.expand(eq.rhs))
    steps.append(f"Expanded: {expanded}")

    simplified = sp.Eq(sp.simplify(expanded.lhs), sp.simplify(expanded.rhs))
    steps.append(f"Simplified: {simplified}")

    solution = sp.solve(simplified, x)
    steps.append(f"Solution: x = {solution}")

    return "\n".join(steps)

def solve_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    text = clean_text(text)
    return text, solve_equation(text)
