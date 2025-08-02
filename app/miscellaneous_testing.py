from parameter import extract_variable_names, create_sympy_parsing_params
from evaluation import Params, evaluation_function, is_equivalent_sympy, extract_symbols, contains_special_math, strip_outer_parens
from sympy.parsing.sympy_parser import parse_expr
from sympy import solve, Eq, simplify, Symbol, integrate, diff, sqrt
from llm_conversion import convert_diff, convert_integral, convert_other
from re_conversion import convert_diff_re, convert_other_re, convert_integral_re
from FMX2_symbols import Fluids

from sympy import symbols, Function, Integral
# f = symbols('f')
# x = symbols('x', real=True)
# print(create_sympy_parsing_params(Params(symbol_assumptions={"x": {"real": True},}),"abs","sqrt(x**2)"))
print(evaluation_function("Du_vec/Dt", "smart_derivative(u_vec,t) + grad(u_vec)*u_vec", Params()))
# fd = Fluids()
# print(simplify(fd.u_vec - fd.u_vec) == 0)
