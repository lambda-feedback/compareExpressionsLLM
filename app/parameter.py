from sympy import Symbol, Function
import re
from typing import Any, TypedDict
from sympy import sympify
from typing import Set

class Params(TypedDict):
    pass

def extract_variable_names(*expressions: str) -> Set[str]:
    """
    Extract variable names from expressions, excluding reserved words.
    """
    pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    reserved = {
        # sympy等の一般関数
        "Abs","sin","cos","tan","sqrt","abs","exp","log",
        "I","E","pi","beta","gamma","zeta","diff","int","Derivative","Integral",
        # あなたの環境の演算子/関数
        "Gradient","Divergence","Curl","dot","cross",
        "smart_derivative","smart_dot"  # ← 追加
    }
    names = set()
    for expr in expressions:
        for m in pattern.findall(expr or ""):
            if m not in reserved and not m.isnumeric():
                names.add(m)
    return names

def parse_function_spec(func_str: str) -> tuple[str, list[str]]:
    """
    "f(x,y)" → ("f", ["x", "y"])
    """
    match = re.match(r'^([a-zA-Z_]\w*)\s*\(([^()]*)\)$', func_str.strip())
    if not match:
        raise ValueError(f"Invalid function specification: {func_str}")
    func_name = match.group(1)
    args = [arg.strip() for arg in match.group(2).split(",") if arg.strip()]
    return func_name, args
def apply_declared_functions(expr: str, func_params: list[str]) -> str:
    # {"y": ["x"], "f": ["t"], "g": ["x"]} を作る
    decl = {}
    for s in func_params:
        # "y(x,z)" → name="y", args=["x","z"]
        name, args = s.split("(", 1)
        name = name.strip()
        args = args.rstrip(")").strip()
        arglist = [a.strip() for a in args.split(",")] if args else []
        decl[name] = arglist

    # 裸の関数名を適用形にする：\bname\b(?!\s*\()
    # 例: y → y(x), f → f(t)
    for name, args in decl.items():
        argstr = ",".join(args)
        expr = re.sub(rf"\b{name}\b(?!\s*\()", f"{name}({argstr})", expr)
    return expr

def _parse_fn(sig: str) -> tuple[str, list[str]]:
    # "y(x,z)" -> ("y", ["x","z"])
    name, args = sig.split("(", 1)
    name = name.strip()
    args = args.rstrip(")").strip()
    argv = [a.strip() for a in args.split(",")] if args else []
    return name, argv

def parse_domain(domain_str: str) -> dict:
    """
    Receive "(a,b]", "[a,b]", "(-2,2)", "(0, pi]", "(-oo, oo)" etc.
    Return {"left": sympy.Expr, "right": sympy.Expr, "left_open": bool, "right_open": bool}
    """
    domain_str = domain_str.strip()
    match = re.match(r'^([\(\[])\s*([^,]+)\s*,\s*([^,]+)\s*([\)\]])$', domain_str)
    if not match:
        raise ValueError(f"Invalid domain format: {domain_str}")

    left_br, left_val, right_val, right_br = match.groups()
    return {
        "left": sympify(left_val),
        "right": sympify(right_val),
        "left_open": (left_br == "("),
        "right_open": (right_br == ")"),
    }

def check_in_domain(value, domain: dict) -> bool:
    left, right = domain["left"], domain["right"]

    if domain["left_open"]:
        if not (value > left):
            return False
    else:
        if not (value >= left):
            return False

    if domain["right_open"]:
        if not (value < right):
            return False
    else:
        if not (value <= right):
            return False

    return True


def create_sympy_parsing_params(params: dict, *expressions: str) -> dict:
    symbol_dict: dict[str, Any] = {}

    # assumptions
    assumptions: dict[str, dict] = params.get("symbol_assumptions", {}) or {}

    # functions
    fn_list = params.get("function") or []
    if isinstance(fn_list, (str, bytes)):
        fn_list = [fn_list]

    for sig in fn_list:
        fn_name, arg_names = _parse_fn(sig)
        symbol_dict[fn_name] = Function(fn_name)
        for a in arg_names:
            if a not in symbol_dict:
                attrs = assumptions.get(a, {})
                symbol_dict[a] = Symbol(a, **attrs) if isinstance(attrs, dict) else Symbol(a)

    for v, attrs in assumptions.items():
        if v not in symbol_dict:
            symbol_dict[v] = Symbol(v, **(attrs if isinstance(attrs, dict) else {}))

    # --- NEW: domain サポート ---
    domain = None
    if "domain" in params:
        domain = parse_domain(params["domain"])

    return {
        "symbol_dict": symbol_dict,
        "domain": domain,
    }

d = parse_domain("(0,3)")
print(check_in_domain(0, d))  # False
print(check_in_domain(3, d))  # False
print(check_in_domain(2.9, d))  # True