import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Any, TypedDict, Union
from sympy import solve, Eq, simplify, Symbol, Function, integrate, diff
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
from parameter import create_sympy_parsing_params, Params
from re_conversion import convert_diff_re, convert_integral_re, convert_other_re


class Result(TypedDict):
    is_correct: bool
    sympy_result: Union[bool, None]
    llm_result: bool
    mismatch_info: str


transformations = standard_transformations + (implicit_multiplication_application,)


def has_unbalanced_parentheses(expr: str) -> bool:
    """
    Check if the expression has unbalanced parentheses
    """
    return expr.count("(") != expr.count(")")
class contains_special_math():
    def contains_diff(self, expr: str) -> bool:
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
            r"partial\([^\)]+\)",
            r"del\([^\)]+\)",
        ]
        return any(re.search(p, expr) for p in patterns)
    def contains_integral(self, expr: str) -> bool:
        patterns = [
            # Basic integrals: int(f,x), integrate(f,x), Integral(f,x)
            r"\bint\(\s*[^,]+,\s*\w+\s*\)",                                 # int(f, x)
            r"\bintegrate\(\s*[^,]+,\s*\w+\s*\)",                          # integrate(f, x)
            r"\bIntegral\(\s*[^,]+,\s*\w+\s*\)",                           # Integral(f, x)
            r"int_[^ \^_]+?\^[^ \^_]+?\([^)]*\)\s*d\w+",
            r"\bint_\w+\([^)]*\)",

            # Circular integrals: int(f,x,circular=True), oint(f,x)
            r"\bint\(\s*[^,]+,\s*\w+,\s*circular\s*=\s*True\s*\)",
            r"\bintegrate\(\s*[^,]+,\s*\w+,\s*circular\s*=\s*True\s*\)",
            r"\bIntegral\(\s*[^,]+,\s*\w+,\s*circular\s*=\s*True\s*\)",
            r"\boint\(\s*[^,]+,\s*\w+\s*\)",

            # Definite integrals: int(f,(x,a,b)), integrate(f,(x,a,b))
            r"\bint\(\s*[^,]+,\s*\(\s*\w+\s*,\s*[^,]+,\s*[^)]+\s*\)\s*\)",
            r"\bintegrate\(\s*[^,]+,\s*\(\s*\w+\s*,\s*[^,]+,\s*[^)]+\s*\)\s*\)",
            r"\bIntegral\(\s*[^,]+,\s*\(\s*\w+\s*,\s*[^,]+,\s*[^)]+\s*\)\s*\)",

            # Lower-limit only integrals: int(f,(x,a))
            r"\bint\(\s*[^,]+,\s*\(\s*\w+\s*,\s*[^)]+\s*\)\s*\)",
            r"\bintegrate\(\s*[^,]+,\s*\(\s*\w+\s*,\s*[^)]+\s*\)\s*\)",
            r"\bIntegral\(\s*[^,]+,\s*\(\s*\w+\s*,\s*[^)]+\s*\)\s*,\s*circular\s*=\s*True\s*\)",

            # Nested integrals: int(int(f,x),y), etc.
            r"\bint\(\s*int\([^)]+\),\s*\w+\s*\)",
            r"\bintegrate\(\s*[^,]+(?:,\s*\w+){2,}\)",                    # integrate(f,x,y,z)
            r"\bIntegral\(\s*[^,]+(?:,\s*\w+){2,}\)",                     # Integral(f,x,y,z)

            # Unicode integral signs
            r"∫", r"∮", r"∫_.*?\^.*?[^a-zA-Z]", r"∫∫", r"∫∫∫",
        ]
        return any(re.search(p, expr) for p in patterns)
    def contains_other(self, expr: str) -> bool:
        patterns = [
            # Summation and delta functions
            r"Σ", r"∑",                                            # summation symbols
            r"Π", r"∏",                                            # product symbols
            r"DiracDelta",                                         #delta functions
            # Infinity variations
            r"Infinity", r"infinity", r"∞", r"oo", r"Inf", r"inf", r"Infty", r"infty",
            r"\w+\s*[·⋅]\s*\w+",        
            r"\bdot\(\s*\w+\s*,\s*\w+\s*\)", 
            r"\b\w+\.dot\(\s*\w+\s*\)",
            # Gradient
            r"∇\w+",
            r"\bgradient\([^)]*\)",
            r"\bgrad\([^)]*\)",

            # Divergence
            r"∇·\w+",
            r"\bdiv\([^)]*\)",
            r"\bdivergence\([^)]*\)",

            # Curl
            r"∇×\w+",
            r"\bcurl\([^)]*\)",
            r"\brot\([^)]*\)",
        ]
        return any(re.search(p, expr) for p in patterns)

def replace_greek_symbols(expr: str) -> str:
    greek_map = {
        # 小文字
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
        "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
        "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
        "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
        "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
        "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
        # 大文字
        "Α": "Alpha", "Β": "Beta", "Γ": "Gamma", "Δ": "Delta",
        "Ε": "Epsilon", "Ζ": "Zeta", "Η": "Eta", "Θ": "Theta",
        "Ι": "Iota", "Κ": "Kappa", "Λ": "Lambda", "Μ": "Mu",
        "Ν": "Nu", "Ξ": "Xi", "Ο": "Omicron", "Π": "Pi",
        "Ρ": "Rho", "Σ": "Sigma", "Τ": "Tau", "Υ": "Upsilon",
        "Φ": "Phi", "Χ": "Chi", "Ψ": "Psi", "Ω": "Omega"
    }

    for greek_letter, ascii_name in greek_map.items():
        expr = expr.replace(greek_letter, ascii_name)
    return expr


def extract_symbols(expr: str) -> dict:

    high_order_pattern = r"\b(?:d|del)\*\*\d+[a-zA-Z_]\w*/(?:d|del)[a-zA-Z_]\w*\*\*\d+\b"
    first_order_pattern = r"\b(?:d|del)[a-zA-Z_]\w*/(?:d|del)[a-zA-Z_]\w*\b"
    material_pattern = r"\bD_[a-zA-Z_]\w*_[a-zA-Z_]\w*\b"

    intg_pattern = r"(?:o?intg)\((?:[^()]+|\((?:[^()]+|\([^()]*\))*\))*\)"

    nabla_pattern = r"(?:\bgrad\b|\bdivg\b|\brot\b|\bdot\b|\bcross\b|\bvec\b|\bhat\b)"

    combined_pattern = f"{high_order_pattern}|{first_order_pattern}|{material_pattern}|{intg_pattern}|{nabla_pattern}"

    matches = re.findall(combined_pattern, expr)

    unique_matches = set(matches)
    symbol_dict = {m: Symbol(m) for m in unique_matches}

    return symbol_dict

def is_equivalent_sympy(expr1, expr2, params) -> Union[bool, None]:
    """
    Return True/False if comparable with SymPy,
    or None if an error occurs.
    """
    if not expr1.strip() and not expr2.strip():
        return True
    if not expr1.strip() or not expr2.strip():
        return False

    try:
        parsing_params = create_sympy_parsing_params(params, expr1, expr2)
        raw_dict = parsing_params["symbol_dict"]
        transformations = parsing_params.get("extra_transformations", standard_transformations + (implicit_multiplication_application,))

        symbols1 = extract_symbols(expr1)
        symbols2 = extract_symbols(expr2)

        local_dict = {
            **symbols1,
            **symbols2,
        }
        local_dict["intg"] = Function("intg")
        local_dict["ointg"] = Function("ointg")
        local_dict["grad"] = Function("grad")
        local_dict["divg"] = Function("divg")
        local_dict["rot"] = Function("rot")
        local_dict["dot"] = Function("dot")
        local_dict["cross"] = Function("cross")
        local_dict["vec"] = Function("vec")
        local_dict["hat"] = Function("hat")

        for name, attrs in raw_dict.items():
            if name == "integrate":
                local_dict[name] = Function(name)
            elif name not in local_dict:
                local_dict[name] = Symbol(name, **attrs) if isinstance(attrs, dict) else attrs

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

        # Parse expressions
        expr1_parsed = parse_expr(expr1, transformations=transformations, local_dict=local_dict)
        expr2_parsed = parse_expr(expr2, transformations=transformations, local_dict=local_dict)

        print("expr1_parsed:", expr1_parsed)
        print("expr2_parsed:", expr2_parsed)

        return simplify(expr1_parsed - expr2_parsed) == 0

    except Exception as e:
        print(f"SymPy error: {e}")
        return False

def evaluation_function(response, answer, params):

    if has_unbalanced_parentheses(response) or has_unbalanced_parentheses(answer):
        return {
            "is_correct": False,
        }
    if response.strip().startswith(("+", "*", "/")) or response.strip().endswith(("+", "-", "*", "/")):
        return {
            "is_correct": False,
        }
    response = response.replace("^", "**")
    answer = answer.replace("^", "**")
    response = replace_greek_symbols(response)
    answer = replace_greek_symbols(answer)
    print(response, answer)

    if response.strip() == "" or answer.strip() == "":
        needs_conversion = False
    else:
        checker = contains_special_math()
        response_has_diff = checker.contains_diff(response)
        response_has_integral = checker.contains_integral(response)
        response_has_other = checker.contains_other(response)

        answer_has_diff = checker.contains_diff(answer)
        answer_has_integral = checker.contains_integral(answer)
        answer_has_other = checker.contains_other(answer)

        needs_conversion = (
            response_has_diff or response_has_integral or response_has_other or
            answer_has_diff or answer_has_integral or answer_has_other
        )

    if needs_conversion:
        if response_has_other:
            response = convert_other_re(response, params) 
        if response_has_diff:
            response = convert_diff_re(response, params) 
        if response_has_integral:
            response = convert_integral_re(response, params) 
        
        if answer_has_other:
            answer = convert_other_re(answer, params) 
        if answer_has_diff:
            answer = convert_diff_re(answer, params) 
        if answer_has_integral:
            answer = convert_integral_re(answer, params) 

        print(response, answer)

    result = is_equivalent_sympy(response, answer, params)

    return {"is_correct": result}