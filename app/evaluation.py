import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Any, TypedDict, Union
from sympy import solve, Eq, simplify, Expr, symbols, Symbol, Function, FunctionClass, Integral, Derivative, Matrix, Abs, sin, cos, tan, sqrt, log, exp, oo
from sympy.core.function import AppliedUndef
from sympy.matrices import MatrixBase
from sympy import Basic
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
from parameter import create_sympy_parsing_params, Params, apply_declared_functions, parse_domain, check_in_domain
from re_conversion import convert_diff_re, convert_integral_re, convert_other_re
from llm_conversion import convert_diff, convert_integral, convert_other
from FMX2_symbols import Fluids
from evaluation_symbolise import symbolise_by_operators, is_matching_parens
from sympy import sympify


class Result(TypedDict):
    is_correct: bool
    sympy_result: Union[bool, None]
    llm_result: bool
    mismatch_info: str

def strip_outer_parens(expr: str) -> str:
    expr = expr.strip()
    if not expr.startswith("(") or not expr.endswith(")"):
        return expr

    depth = 0
    for i, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth == 0 and i != len(expr) - 1:
            return expr
    return expr[1:-1].strip()


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

        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
        "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
        "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
        "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
        "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
        "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",

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

    # high_order_pattern = r"\b(?:d|del)\*\*\d+[a-zA-Z_]\w*/(?:d|del)[a-zA-Z_]\w*\*\*\d+\b"
    # first_order_pattern = r"\b(?:d|del)[a-zA-Z_]\w*/(?:d|del)[a-zA-Z_]\w*\b"
    material_pattern = (
        r"\bD_[A-Za-z_]\w*_[A-Za-z_]\w*\b"          # D_var1_var2 形式
        r"|"                                        # または
        r"\bD[A-Za-z_]\w*/D[A-Za-z_]\w*\b"          # Dy/Dx 形式
        r"|"                                        # または
        r"\bD/D[A-Za-z_]\w*(?:\(\s*[A-Za-z_]\w*\s*\)|\s+[A-Za-z_]\w*)"  # D/Dx(y) 形式
    )
    # intg_pattern = r"(?:o?intg)\((?:[^()]+|\((?:[^()]+|\([^()]*\))*\))*\)"

    # nabla_pattern = r"(?:\bgrad\b|\bdivg\b|\brot\b|\bdot\b|\bcross\b|\bvec\b|\bhat\b)"

    combined_pattern = f"{material_pattern}"

    matches = re.findall(combined_pattern, expr)

    unique_matches = set(matches)
    symbol_dict = {m: Symbol(m) for m in unique_matches}

    return symbol_dict

def is_equivalent_sympy(expr1, expr2, params) -> Union[bool, None]:
    """
    Return True/False if comparable with SymPy,
    or False if an error occurs.
    """

    if isinstance(expr1, str) and isinstance(expr2, str):
        if not expr1.strip() and not expr2.strip():
            return True
        if not expr1.strip() or not expr2.strip():
            return False

    try:
        # Always convert to string before parsing assumptions
        parsing_params = create_sympy_parsing_params(params, str(expr1), str(expr2))
        raw_dict = parsing_params["symbol_dict"]
        transformations = parsing_params.get(
            "extra_transformations",
            standard_transformations + (implicit_multiplication_application,)
        )

        # Optional: Extract symbols from string inputs
        # symbols1 = extract_symbols(expr1) if isinstance(expr1, str) else expr1
        # symbols2 = extract_symbols(expr2) if isinstance(expr2, str) else expr2

        # Optional fluid context
        fd = Fluids()
        fd_dict = vars(fd)
        valid_types = (Basic, MatrixBase)
        fd_func_dict = {
            k[:-5]: v
            for k, v in fd_dict.items()
            if k.endswith('_func') and isinstance(v, FunctionClass)
        }
        func_names = set(fd_func_dict.keys())
        fd_filtered = {
            k: v for k, v in fd_dict.items()
            if isinstance(v, valid_types)
            and not k.endswith('_func')
            and not isinstance(v, AppliedUndef)
            and k not in func_names
        }

        # Build local_dict for parser
        local_dict = {
            "Gradient": fd.Gradient,
            "Divergence": fd.Divergence,
            "Curl": fd.Curl,
            "smart_derivative": fd.smart_derivative,
            "smart_dot": fd.smart_dot,
            **fd_func_dict,
            **fd_filtered,
            # **symbols1,
            # **symbols2,
        }

        for name, sym in raw_dict.items():
            if isinstance(sym, dict):
                local_dict[name] = Symbol(name, **sym)
            else:
                if params:
                    if isinstance(params, dict) and params:
                        local_dict[name] = sym
                    elif name not in local_dict:
                        local_dict[name] = sym

        if isinstance(params, dict) and params.get("function"):
            if isinstance(expr1, str):
                expr1 = apply_declared_functions(expr1, params["function"])
            if isinstance(expr2, str):
                expr2 = apply_declared_functions(expr2, params["function"])

        def ensure_expr(expr):
            if isinstance(expr, str):
                func_args_map = {
                    "u": (fd.x, fd.y, fd.z, fd.t),
                    "v": (fd.x, fd.y, fd.z, fd.t),
                    "w": (fd.x, fd.y, fd.z, fd.t),
                    "T": (fd.x, fd.y, fd.z, fd.t),
                    "rho": (fd.x, fd.y, fd.z, fd.t),
                    "p": (fd.x, fd.y, fd.z, fd.t),
                    "u_r": (fd.r, fd.theta, fd.z, fd.t),
                    "u_theta": (fd.r, fd.theta, fd.z, fd.t),
                    "u_z": (fd.r, fd.theta, fd.z, fd.t),
                }

                applied_funcs = {}
                for name, func in local_dict.items():
                    if isinstance(func, FunctionClass) and name in func_args_map:
                        args = func_args_map[name]
                        applied_funcs[name] = func(*args)

                for name, applied in applied_funcs.items():
                    pattern = rf'(?<!\w){name}(?!\w|\s*\()'
                    expr = re.sub(pattern, f'({str(applied)})', expr)

                return parse_expr(expr, transformations=transformations, local_dict=local_dict)
            else:
                return expr

        # Handle equations
        if "=" in str(expr1) and "=" in str(expr2):
            def unwrap_parens_if_equal(s: str) -> str:
                            s = s.strip()
                            if "=" not in s:
                                if s.startswith("(") and s.endswith(")") and is_matching_parens(s):
                                    inner = s[1:-1].strip()
                                    return inner
                            else:
                                left, right = s.split("=")
                                if left.startswith("(") and left.endswith(")") and is_matching_parens(left):
                                    left = left[1:-1].strip()
                                if right.startswith("(") and right.endswith(")") and is_matching_parens(right):
                                    right = right[1:-1].strip()
                                return left+"="+right
                            return s
            if isinstance(expr1, str) and isinstance(expr2, str):
                expr1 = unwrap_parens_if_equal(expr1)
                expr2 = unwrap_parens_if_equal(expr2)
            lhs1, rhs1 = str(expr1).split("=")
            lhs2, rhs2 = str(expr2).split("=")

            lhs1_parsed = ensure_expr(lhs1)
            rhs1_parsed = ensure_expr(rhs1)
            lhs2_parsed = ensure_expr(lhs2)
            rhs2_parsed = ensure_expr(rhs2)

            eq1 = Eq(lhs1_parsed - rhs1_parsed, 0)
            eq2 = Eq(lhs2_parsed - rhs2_parsed, 0)

            all_symbols = eq1.free_symbols.union(eq2.free_symbols)
            sol1 = solve(eq1, list(all_symbols))
            sol2 = solve(eq2, list(all_symbols))

            return set(sol1) == set(sol2)

        # Handle expression comparison
        expr1_parsed = ensure_expr(expr1)
        expr2_parsed = ensure_expr(expr2)
        expr1 = apply_declared_functions(expr1, params.get("function", []))
        expr2 = apply_declared_functions(expr2, params.get("function", []))


        if isinstance(expr1_parsed, MatrixBase) and isinstance(expr2_parsed, MatrixBase):
            return simplify(expr1_parsed - expr2_parsed) == Matrix.zeros(*expr1_parsed.shape)
        else:
            return simplify(expr1_parsed - expr2_parsed) == 0


    except Exception as e:
        if "could not solve" in str(e).lower():
            try:
                expr1_symbolised, expr2_symbolised, symbol_map = symbolise_by_operators(expr1, expr2)
                if "=" in str(expr1) and "=" in str(expr2):
                    lhs1, rhs1 = str(expr1_symbolised).split("=")
                    lhs2, rhs2 = str(expr2_symbolised).split("=")
                    lhs1_expr = parse_expr(lhs1.strip())
                    rhs1_expr = parse_expr(rhs1.strip())
                    lhs2_expr = parse_expr(lhs2.strip())
                    rhs2_expr = parse_expr(rhs2.strip())
                    eq1 = Eq(lhs1_expr - rhs1_expr, 0)
                    eq2 = Eq(lhs2_expr - rhs2_expr, 0)
                    all_symbols = eq1.free_symbols.union(eq2.free_symbols)
                    sol1 = solve(eq1, list(all_symbols))
                    sol2 = solve(eq2, list(all_symbols))
                    return set(sol1) == set(sol2)
                else:
                    return simplify(expr1_symbolised - expr2_symbolised) == 0
            except Exception as fallback_error:
                print(f"Fallback error: {fallback_error}")
                return None
        else:
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
    response = response.replace(" ", "")
    answer = answer.replace(" ", "")
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
            # response = convert_other(response, params).content.strip()
            response = convert_other_re(response, params) 
        if response_has_diff:
            # response = convert_diff(response, params).content.strip()
            response = convert_diff_re(response, params) 
        if response_has_integral:
            # response = convert_integral(response, params).content.strip()
            response = convert_integral_re(response, params) 
        
        if answer_has_other:
            # answer = convert_other(answer, params).content.strip()
            answer = convert_other_re(answer, params) 
        if answer_has_diff:
            # answer = convert_diff(answer, params).content.strip()
            answer = convert_diff_re(answer, params) 
        if answer_has_integral:
            # answer = convert_integral(answer, params).content.strip()
            answer = convert_integral_re(answer, params) 

        print("response converted: " + response)
        print("answer converted: " + answer)
    response = strip_outer_parens(response)
    answer = strip_outer_parens(answer)
    result = is_equivalent_sympy(response, answer, params)

    if "domain" in params and params["domain"]:
        try:
            domain = params["domain"]
            if isinstance(domain, str):
                domain = parse_domain(domain)

            val = sympify(response)

            if val.is_number:
                if not check_in_domain(val, domain):
                    return {"is_correct": False}

        except Exception as e:
            print("error in domain check:", e)

    return {"is_correct": result}