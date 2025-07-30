from parameter import extract_variable_names, create_sympy_parsing_params
from evaluation import Params, evaluation_function, is_equivalent_sympy, extract_symbols, contains_special_math
from sympy.parsing.sympy_parser import parse_expr
from sympy import solve, Eq, simplify, Symbol, integrate, diff
from llm_conversion import convert_diff, convert_integral, convert_other
from re_conversion import convert_diff_re, convert_other_re, convert_integral_re
import re

# print(extract_derivative_symbols("d^2y/dx^2 + dy/dx + y = 0"))
# print(is_equivalent_sympy("a.dot(b)", "c.dot(d)",Params()))
# print(convert_to_sympy("diff(y,x)", Params()).content.strip())
# print(evaluation_function("∮_{A_sys} rho*u·n dA", "Integral(rho * u.dot(n), (A, A_sys), circular=True)", Params()))
print(convert_other_re("div(u)", Params()))
# print(convert_integral_re(convert_other_re("Integral(rho * u.dot(n), (A, A_sys), circular=True)", Params()), Params()))
# checker = contains_special_math()
# print(checker.contains_integral("∮_{A_sys} rho*u·n dA"))
# print(checker.contains_diff("∮_{A_sys} rho*u·n dA"))
# print(checker.contains_other("∮_{A_sys} rho*u·n dA"))
# test_cases = [
#     # --- 閉曲線積分（∮型） ---
#     ("∮_{a} f dx",                    "ointg(f,x)"),
#     ("∮_{a{a+b}} h dx",                "ointg(h,x)"),
#     ("∮_(z^2) g dx",                  "ointg(g,x)"),
#     ("∮_{A_sys} rho*dot(u,n) dA", "ointg(rho*dot(u,n),A)"),    

#     # --- oint((f,x)) 型 ---
#     ("oint((f,x))",                   "ointg(f,x)"),
#     ("oint({f,x})",                   "ointg(f,x)"),
#     ("oint([f,x])",                   "ointg(f,x)"),

#     # --- int(f,(x,a),circular=True) ---
#     ("int(f,(x,a),circular=True)",    "ointg(f,x)"),
#     ("int(g,[x,a],circular=True)",    "ointg(g,x)"),

#     # --- 定積分 ∫ₐᵇ ---
#     ("∫_a^b f dx",                    "intg(f,(x,a,b))"),
#     ("int(f,(x,a,b))",                "intg(f,(x,a,b))"),

#     # --- 片側積分 ∫ₐ ---
#     ("∫_a f dx",                      "intg(f,x)"),
#     ("int((f,x))",                    "intg(f,x)"),
#     ("int({f,x})",                    "intg(f,x)"),

#     # --- 通常積分 ---
#     ("∫ f dx",                        "intg(f,x)"),
#     ("int(f,x)",                      "intg(f,x)"),
#     ("integrate(f,x)",               "intg(f,x)"),
#     ("Integral(f,x)",                "intg(f,x)"),

#     # --- ネスト ---
#     ("int(int(f,x),y)",              "intg(intg(f,x),y)"),
#     ("int(int(int(h,z),y),x)",       "intg(intg(intg(h,z),y),x)"),
#     ("oint(oint(f,x),y)",              "ointg(ointg(f,x),y)"),
# ]

# from pprint import pprint
# def run_tests():
#     failed = []
#     print("🧪 Running test cases...\n")
#     for expr, expected in test_cases:
#         output = convert_integral_re(expr,Params())
#         if output != expected:
#             print(f"❌ FAIL:")
#             print(f"  Input:    {expr}")
#             print(f"  Expected: {expected}")
#             print(f"  Got:      {output}\n")
#             failed.append((expr, expected, output))
#         else:
#             print(f"✅ PASS:")
#             print(f"  Input:    {expr}")
#             print(f"  Output:   {output}\n")
# run_tests()
