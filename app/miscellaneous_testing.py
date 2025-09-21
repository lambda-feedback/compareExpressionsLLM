# from parameter import extract_variable_names, create_sympy_parsing_params
# from evaluation import Params, evaluation_function, is_equivalent_sympy
# from sympy.parsing.sympy_parser import parse_expr
# from sympy import solve, Eq, simplify, Symbol, integrate, diff, sqrt, oo
# from llm_conversion import convert_diff, convert_integral, convert_other
# from re_conversion import convert_diff_re, convert_other_re, convert_integral_re
# from FMX2_symbols import Fluids
# from sympy import symbols, Function, Integral
import re

# ["Drho/Dt=-div(u_vec)", "Drho/Dt+div(u_vec)=0", Params(), True]
# print(evaluation_function("grad(u_r)=delu_r/delr*hat(r)+delu_r/deltheta*hat(theta)+delu_r/delz*hat(z)","grad(u_r)=delu_r/delr*hat(r)+delu_r/deltheta*hat(theta)+delu_r/delz*hat(z)", Params()))
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

transformations = standard_transformations + (implicit_multiplication_application,)
expr = parse_expr("log(xy)", transformations=transformations)
print(expr)