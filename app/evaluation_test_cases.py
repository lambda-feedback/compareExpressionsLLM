from evaluation import Params, evaluation_function
from parameter import create_sympy_parsing_params
# [response, answer, params, expected]
test_cases = [
            ["2+2", "4", Params(), True], #1
            ["sin(x)**2 + cos(x)**2", "1", Params(), True],
            ["x+y", "y+x", Params(), True],
            ["x*y", "x+y", Params(), False],
            ["x**2 + 2*x + 1", "(x+1)**2", Params(), True],
            ["x**2 - 1", "(x-1)*(x+1)", Params(), True],
            ["x^5-1", "(x-1)*(x**4+x**3+x**2+x+1)", Params(), True],
            ["sin(x) + cos(x)", "cos(x) + sin(x)", Params(), True],
            ["sin(x) * cos(x)", "sin(x) + cos(x)", Params(), False],
            ["exp(x) * exp(y)", "exp(x+y)", Params(), True],
            ["log(x*y)", "log(x) + log(y)", Params(), False],
            ["x**3 + x**2", "x**2 * (x + 1)", Params(), True],
            ["", "", Params(), True],       
            ["", "x", Params(), False],
            ["1+", "1", Params(), False],
            ["x+1=0", "-2x-2=0", Params(), True],
            ["dy/dx", "diff(y, x)", Params(), True],
            ["(x+y)/x", "1 + y/x", Params(), True],
            ["∂y/∂x", "diff(y, x)", Params(), True],
            ["∫f(x)dx", "int(f(x), x)", Params(), True],
            ["∂^2y/∂x^2", "diff(diff(y, x), x)", Params(), True],
            ["dy/dx + 1", "diff(y, x) + 1", Params(), True],
            ["∂y/∂x + 1", "diff(y, x) + 1", Params(), True],
            ["dp/dt", "diff(p, t)", Params(), True],
            ["dg/dm", "diff(y,x)", Params(), False],
            ]
test_cases2 = [
            ["infty", "Infinity", Params(), True], #1
            ["sqrt(-1)", "I", Params(), True],
            ["sqrt(x**2)", "x", Params(), False],
            ["1/(x-1)", "1/(1-x)", Params(), False],
            ["x^2", "x**2", Params(), True],
            ["x^^2", "x**2", Params(), False],
            ["d^3y/dx^3", "diff(y, x, x, x)", Params(), True],
            ["∫∫f(x)dxdy", "int(int(f(x), x), y)", Params(), True],
            ["f(x)=x+1", "f(x)-x-1=0", Params(), True],
            ["f(x) = x**2", "f(y) = y**2", Params(), False],#should this always be false?
            ["diff(y,x)+", "diff(y,x)+0", Params(), False],
            ["d/dx(y", "diff(y, x)", Params(), False],
            ["DiracDelta(x)", "0", Params(), False],
            ["∫_{V_sys} ∂ρ/∂t dV", "int(partial(ρ)/partial(t), (V, V_sys))", Params(), True],
            ["∮_{A_sys} ρu·n̂ dA", "Integral(rho * u.dot(n), (A, A_sys), circular=True)", Params(), True],
            ["rho", "ρ", Params(), True],
            ["Dx/Dt=-div(u)", "Dx/Dt+div(u)=0", Params(), True],
            ["(1/rho)*Drho/Dt=-div(u)", "(1/ρ)*Dρ/Dt+div(u)=0", Params(), True],
            ]
test_cases3 = [
            ["abs(x)", "sqrt(x**2)", Params(), False],
            ["abs(x)", "sqrt(x**2)", Params(symbol_assumptions={"x": {"real": True},}), True],
            ]