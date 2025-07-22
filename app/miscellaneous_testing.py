from parameter import extract_variable_names, create_sympy_parsing_params
from evaluation import Params, evaluation_function
from sympy.parsing.sympy_parser import parse_expr
from sympy import solve, Eq, simplify, Symbol

# parsing_params = create_sympy_parsing_params(Params(symbol_assumptions={"x": {"real": True}}), "x+2+z", "x*y*z")
# local_dict = parsing_params["symbol_dict"]
# print(local_dict)
lhs, rhs = "2*x+1", "0"
eq = Eq(parse_expr(lhs), parse_expr(rhs))
print(solve(eq))
lhs, rhs = "x", "-1/2"
eq = Eq(parse_expr(lhs), parse_expr(rhs))
print(solve(eq))
