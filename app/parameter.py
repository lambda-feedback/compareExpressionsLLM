from sympy import Symbol, Function, MatrixSymbol, IndexedBase
import re
from typing import Any, TypedDict
from sympy import Eq, Ne, sympify
from typing import Any, TypedDict, Set, Tuple, List, Dict

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

def parse_function_spec(func_str: str) -> Tuple[str, List[str]]:
    """
    "f(x,y)" → ("f", ["x", "y"])
    """
    match = re.match(r'^([a-zA-Z_]\w*)\s*\(([^()]*)\)$', func_str.strip())
    if not match:
        raise ValueError(f"Invalid function specification: {func_str}")
    func_name = match.group(1)
    args = [arg.strip() for arg in match.group(2).split(",") if arg.strip()]
    return func_name, args
def apply_declared_functions(expr: str, func_params: List[str]) -> str:
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

def _parse_fn(sig: str) -> Tuple[str, List[str]]:
    # "y(x,z)" -> ("y", ["x","z"])
    name, args = sig.split("(", 1)
    name = name.strip()
    args = args.rstrip(")").strip()
    argv = [a.strip() for a in args.split(",")] if args else []
    return name, argv

def parse_domain(domain_str: str) -> Dict:
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
    """
    Check if a value satisfies the given domain.
    """
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


def parse_constraint(expr: str):
    """
    Convert a constraint string like 'x > 0', 'a + b = 1', 'm != n' into a SymPy object.
    """
    expr = expr.strip()

    # = を Eq に変換
    if "=" in expr and "==" not in expr and "!=" not in expr:
        left, right = expr.split("=", 1)
        return Eq(sympify(left), sympify(right))

    # != を Ne に変換
    if "!=" in expr:
        left, right = expr.split("!=", 1)
        return Ne(sympify(left), sympify(right))

    # >, <, >=, <= は sympify が処理可能
    return sympify(expr)


def create_sympy_parsing_params(params: Dict, *expressions: str) -> Dict:
    symbol_dict: Dict[str, Any] = {}

    # assumptions
    assumptions: Dict[str, dict] = params.get("symbol_assumptions", {}) or {}

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

    # --- domains ---
    domains: Dict[str, dict] = {}
    if "domains" in params:
        for var, dstr in params["domains"].items():
            domains[var] = parse_domain(dstr)

    # --- function properties ---
    fn_properties: Dict[str, Dict[str, Any]] = {}
    if "function_properties" in params:
        for sig, props in params["function_properties"].items():
            fn_name, args = _parse_fn(sig)
            fn_properties[sig] = props

    # --- object types ---
    object_types: Dict[str, str] = {}
    if "object_type" in params:
        for var, otype in params["object_type"].items():
            object_types[var] = otype
            if otype == "vector":
                symbol_dict[var] = IndexedBase(var)
            elif otype == "matrix":
                symbol_dict[var] = MatrixSymbol(var, 3, 3)
            elif otype == "scalar":
                symbol_dict[var] = Symbol(var)
            elif otype == "tensor":
                symbol_dict[var] = IndexedBase(var)

    # --- NEW: constraints ---
    constraints = []
    if "constraints" in params:
        for c in params["constraints"]:
            try:
                constraints.append(parse_constraint(c))
            except Exception as e:
                raise ValueError(f"Invalid constraint: {c}") from e

    return {
        "symbol_dict": symbol_dict,
        "domains": domains,
        "function_properties": fn_properties,
        "object_types": object_types,
        "constraints": constraints,
    }

def validate_constraints(values: Dict[str, Any], parsed: Dict) -> Dict[str, Any]:
    results = {"ok": True, "violations": []}

    # --- ドメインチェック ---
    for var, domain in parsed.get("domains", {}).items():
        if var in values and isinstance(values[var], (int, float)):
            val = values[var]
            if not check_in_domain(val, domain):
                results["ok"] = False
                results["violations"].append(f"{var}={val} not in {domain}")

    # --- 制約チェック ---
    for cons in parsed.get("constraints", []):
        try:
            # list/tuple は無視して代入しない
            safe_values = {k: v for k, v in values.items() if not isinstance(v, (list, tuple))}
            expr = cons.subs(safe_values)

            if isinstance(cons, Eq):
                if not bool(expr):
                    results["ok"] = False
                    results["violations"].append(f"Constraint {cons} not satisfied with {safe_values}")
            elif isinstance(cons, Ne):
                if not bool(expr):
                    results["ok"] = False
                    results["violations"].append(f"Constraint {cons} not satisfied with {safe_values}")
            else:
                if not bool(expr):
                    results["ok"] = False
                    results["violations"].append(f"Constraint {cons} not satisfied with {safe_values}")
        except Exception as e:
            results["ok"] = False
            results["violations"].append(f"Constraint check failed for {cons}: {e}")

    # --- オブジェクト型チェック ---
    for var, otype in parsed.get("object_types", {}).items():
        if var in values:
            val = values[var]
            if otype == "vector" and not isinstance(val, (list, tuple)):
                results["violations"].append(f"{var} should be a vector but got {val}")
                results["ok"] = False
            if otype == "matrix" and not (hasattr(val, "shape") or isinstance(val, list)):
                results["violations"].append(f"{var} should be a matrix but got {val}")
                results["ok"] = False

    return results

params = {
    "symbol_assumptions": {"x": {"real": True}, "y": {"real": True}},
    "domains": {"x": "(0,3)", "y": "[-1,1]"},
    "constraints": ["x + y = 1", "x > 0"],
    "object_type": {"u": "vector"}
}

parsed = create_sympy_parsing_params(params)

# 検証
print(validate_constraints({"x": 2, "y": -1, "u": [1,2,3]}, parsed))
# -> {'ok': True, 'violations': []}

print(validate_constraints({"x": -1, "y": 2, "u": 5}, parsed))
# -> {'ok': False, 'violations': [...違反内容...]}