from sympy import Symbol, Function
import re

# default_params = {
#     "symbol_assumptions": {}
# }


# def get_default_params() -> dict:
#     """
#     Return a copy of the default parameters.
#     """
#     return default_params.copy()

def extract_variable_names(*expressions: str) -> set:
    """
    Extract variable names from expressions, excluding reserved words.
    """
    pattern = re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b')
    reserved = {"sin", "cos", "tan", "sqrt", "abs", "exp", "log",
                "I", "E", "pi", "beta", "gamma", "zeta", "diff", "int"}
    return [
        m for expr in expressions for m in pattern.findall(expr)
        if m not in reserved and not m.isnumeric()
    ]

def create_sympy_parsing_params(params, *expressions):
    """
    Build parsing parameters for SymPy:
    - Default: all variables are complex symbols.
    - symbol_assumptions: override attributes for specific symbols.
    """
    symbol_dict = {}

    # 1️⃣ get symbol_assumptions (default: empty dict)
    assumptions = params.get("symbol_assumptions", {})

    # 2️⃣ extract all variables from expressions
    unknown_vars = extract_variable_names(*expressions)

    # 3️⃣ for each variable, register it with appropriate attributes
    for v in unknown_vars:
        if v in assumptions:
            # if in assumptions: use specified attributes
            try:
                sym = Symbol(v, **assumptions[v])
                symbol_dict[v] = sym.assumptions0
            except Exception as e:
                raise ValueError(f"Invalid attributes for symbol '{v}': {e}")
        else:
            sym = Symbol(v, complex=True)
            symbol_dict[v] = sym.assumptions0

    return {"symbol_dict": symbol_dict}
