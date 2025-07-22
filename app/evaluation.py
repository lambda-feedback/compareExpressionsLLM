import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from typing import Any, TypedDict
from sympy import solve, Eq, simplify, Symbol
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
from parameter import create_sympy_parsing_params

class Params(TypedDict):
    pass


class Result(TypedDict):
    is_correct: bool
    sympy_result: bool | None
    llm_result: bool
    mismatch_info: str


transformations = standard_transformations + (implicit_multiplication_application,)

def has_unbalanced_parentheses(expr: str) -> bool:
    """
    Check if the expression has unbalanced parentheses
    """
    return expr.count("(") != expr.count(")")

def contains_special_math(expr: str) -> bool:
    """
    Check if the expression contains special mathematical symbols or operations
    """

    patterns = [
        # Differentiation
        r"d(\*\*)?\d*\w*/d\w+(\*\*)?\d*",                     # dy/dx, d**2y/dx**2
        r"d/d\w+\(.*\)",                                       # d/dx(y)
        r"d(\*\*)?\d*/d\w+(\*\*)?\d*\([^\)]+\)",               # d**2/dx**2(y)
        r"D(\*\*)?\d*\w*/D\w+(\*\*)?\d*",                      # Dy/Dx, D**2y/Dx**2
        r"D/D\w+\(.*\)",                                       # D/Dx(y)
        r"∂(\*\*)?\d*\w*/∂\w+(\*\*)?\d*",                      # ∂y/∂x
        r"∂/∂\w+\(.*\)",                                       # ∂/∂x(y)
        r"diff\([^\)]+\)",                                     # diff(y, x), diff(y,x,x)
        # Integration
        r"int\([^\)]+\)",                                      # int(f(x), x)
        r"∫", r"∮",                                            # ∫f(x)dx, ∮f(x)dx
        # Summation and delta functions
        r"Σ", r"∑",                                            # summation symbols
        r"Π", r"∏",                                            # product symbols
        r"DiracDelta",                                         #delta functions
        # Infinity variations
        r"Infinity", r"infinity", r"∞", r"oo", r"Inf", r"inf", r"Infty", r"infty"
    ]
    return any(re.search(p, expr) for p in patterns)

def replace_greek_symbols(expr: str) -> str:
    greek_map = {
        # 小文字
        "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
        "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
        "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
        "nu": "ν", "xi": "ξ", "omicron": "ο", "pi": "π",
        "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
        "phi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
        # 大文字
        "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ",
        "Epsilon": "Ε", "Zeta": "Ζ", "Eta": "Η", "Theta": "Θ",
        "Iota": "Ι", "Kappa": "Κ", "Lambda": "Λ", "Mu": "Μ",
        "Nu": "Ν", "Xi": "Ξ", "Omicron": "Ο", "Pi": "Π",
        "Rho": "Ρ", "Sigma": "Σ", "Tau": "Τ", "Upsilon": "Υ",
        "Phi": "Φ", "Chi": "Χ", "Psi": "Ψ", "Omega": "Ω"
    }

    for ascii_name, greek_letter in greek_map.items():
        expr = re.sub(rf"\b{ascii_name}\b", greek_letter, expr)
    return expr

def is_equivalent_sympy(expr1, expr2, params) -> bool | None:
    """
    Return True/False if comparable with SymPy,
    or None if an error occurs.
    """
    if not expr1.strip() and not expr2.strip():
        return True
    if not expr1.strip() or not expr2.strip():
        return False

    try:
        # Create parsing parameters (expressions渡す版)
        parsing_params = create_sympy_parsing_params(params, expr1, expr2)
        raw_dict = parsing_params["symbol_dict"]
        transformations = parsing_params.get("extra_transformations", ())

        # assumptions0 の辞書なら Symbol を作り直す
        local_dict = {
            name: Symbol(name, **attrs) if isinstance(attrs, dict) else attrs
            for name, attrs in raw_dict.items()
        }

        # Compare with Eq() for equations
        if "=" in expr1 and "=" in expr2:
            lhs1, rhs1 = expr1.split("=")
            lhs2, rhs2 = expr2.split("=")

            lhs1_parsed = parse_expr(lhs1, transformations=transformations, local_dict=local_dict)
            rhs1_parsed = parse_expr(rhs1, transformations=transformations, local_dict=local_dict)
            lhs2_parsed = parse_expr(lhs2, transformations=transformations, local_dict=local_dict)
            rhs2_parsed = parse_expr(rhs2, transformations=transformations, local_dict=local_dict)

            eq1 = Eq(lhs1_parsed - rhs1_parsed, 0)
            eq2 = Eq(lhs2_parsed - rhs2_parsed, 0)

            all_symbols = eq1.free_symbols.union(eq2.free_symbols)

            sol1 = solve(eq1, list(all_symbols))
            sol2 = solve(eq2, list(all_symbols))

            return set(sol1) == set(sol2)
        else:
            expr1_parsed = parse_expr(expr1, transformations=transformations, local_dict=local_dict)
            expr2_parsed = parse_expr(expr2, transformations=transformations, local_dict=local_dict)
            return simplify(expr1_parsed - expr2_parsed) == 0

    except Exception as e:
        print(f"SymPy error: {e}")
        return None

def convert_to_sympy(expr: str, params: Params) -> str:
    load_dotenv()
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    prompt = fr"""
Follow these steps carefully:
A student response and an answer are provided below. Convert the student response into a SymPy expression.

When the following notations in (a) and (b) are used, they must be replaced with the equivalent SymPy expressions.
All the notations in the same square brackets are equivalent, and must be replaced with the notation after the right arrow (->) after the square brackets.

(a) The following notations for derivatives, partial derivatives, and integrals **must be considered strictly equivalent** within the same group:
- [dy/dx, d/dx(y), diff(y,x)] -> diff(y,x)
- [d^2y/dx^2, d**2y/dx**2, diff(y,x,x)] -> diff(y,x,x)
- [d^3y/dx^3, d**3y/dx**3, diff(y,x,x,x)] -> diff(y,x,x,x)
- [Dy/Dx, D/Dx(y)] -> diff(y,t)+v.dot(gradient(y))
- [∂y/∂x, ∂/∂x(y), diff(y,x), partial(y)/partial(x)] -> diff(y,x)
- [∂^2y/∂x^2, ∂**2y/dx**2, diff(y,x,x), partial**2(y)/partial(x)**2, partial^2(y)/partial(x)^2] -> diff(y,x,x)
- [∫f(x)dx, int(f(x),x), integrate(f(x),x), Integral(f(x),x)] -> integrate(f(x), x)
- [∮f(x)dx, int(f(x),x,circular=True), integrate(f(x),x,circular=True), Integral(f(x),x,circular=True)] -> integrate(f(x), x)
- [∫ₐᵇf(x)dx, ∫_a^bf(x)dx, int_a^bf(x)dx, int(f(x),(x,a,b)), integrate(f(x),(x,a,b)), Integral(f(x),(x,a,b))] -> integrate(f(x), (x, a, b))
- [∫∫f(x,y)dxdy, int(int(f(x,y),x),y), integrate(f(x,y),x,y), Integral(f(x,y),x,y)] -> integrate(integrate(f(x,y),x),y)
- [∇f, gradient(f), grad(f)] -> gradient(f)
- [∇·F, div(F), divergence(F)] -> div(f)
- [∇×F, curl(F), rot(F)] -> curl(f)

(b) Other notations that **must be considered equivalent** within the same group:
- [Infinity, infinity, ∞, oo, Inf, inf, Infty, infty] -> oo
- [a·b, a⋅b, a.b, dot(a, b), a.dot(b)] -> a.dot(b)
  *Note: a.b is only equivalent to these if a and b are variables, not constants like 0, 1, π, etc.*
- [a×b, cross(a, b), a.cross(b)] -> a.cross(b)
- [\vec{{a}}, vector(a), a.vector(), Matrix(a)] -> Matrix(a)
- [â, \hat{{a}}, unit(a), normalize(a), a_hat] -> a/Abs(a)
- [exp(x), e**x, e**x, exponential(x)] -> exp(x)

When comparing integrals, assume that any derivative or expression between the integral sign and the differential (e.g., ∂y/∂x in ∫_a ∂y/∂x dx) is the complete integrand, even if parentheses around the integrand are missing.

**Notations from different groups or not listed above are NOT equivalent.**

This is the student response: {expr}
Now convert it to a SymPy expression. Ouput only the SymPy expression, without any additional text or explanation.
    """
    llm_response = llm.invoke(prompt)
    return llm_response

def evaluation_function(response, answer, params):

    if has_unbalanced_parentheses(response) or has_unbalanced_parentheses(answer):
        return {
            "is_correct": False,
            "sympy_result": None,
            "llm_result": False,
            "mismatch_info": "Invalid syntax: unbalanced parentheses"
        }
    response = response.replace("^", "**")
    answer = answer.replace("^", "**")
    response = response.replace(" ", "")
    answer = answer.replace(" ", "")
    response = replace_greek_symbols(response)
    answer = replace_greek_symbols(answer)

    if response.strip() == "" or answer.strip() == "":
        needs_conversion = False
    else:
        needs_conversion = contains_special_math(response) or contains_special_math(answer)

    if needs_conversion:
        response = convert_to_sympy(response, params).content.strip()
        answer = convert_to_sympy(answer, params).content.strip()
    result = None
    result = is_equivalent_sympy(response, answer, params)

    return {"is_correct": result}