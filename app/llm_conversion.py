import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from parameter import Params

def convert_diff(expr: str, params: Params) -> str:
    load_dotenv()
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    prompt = fr"""
Follow these steps carefully:
A response is provided below. Convert it into an expression under the following rules.

When the following notations are found in the response, they must be replaced with the notation after the right arrow (->) after the square brackets.

---

- [dy/dx, d/dx(y), diff(y,x)] -> d_y_x
**Pay special attention: diff(y,x) must always be converted to dydx, but ONLY if the variable names are y and x.**
For example:  
✅ diff(g,m) → d_g_m  
❌ diff(g,m) → d_y_x (**WRONG**)

- [d^2y/dx^2, d**2y/dx**2, diff(y,x,x)] -> d2_y_x
- [d^3y/dx^3, d**3y/dx**3, diff(y,x,x,x)] -> d3_y_x
- [∂y/∂x, ∂/∂x(y), partial(y)/partial(x)] -> del_y_x
**Note: diff(y,x) is NOT equivalent to del_y_x.**
**Note: it must be in the form of del_[]_[] where the variable names are in the square brackets, and nothing else.**
- [∂**2y/dx**2, diff(y,x,x), partial**2(y)/partial(x)**2] -> del2_y_x
- [Dy/Dx, D/Dx(y)] -> (del_y_t+dot(v,grad(y)))
Note that the first variable (in this case "y") always appears after "del_" and "grad(" whereas the second variable (in this case "x") is ignored.
**Note: "diff(y,x)","dy/dx","d_y_x" and "∂y/∂x","del_y_x" are NOT equivalent to "Dy/Dx" or "D/Dx(y)" or "(del_y_t+dot(v,grad(y)))".**
Example: "Drho/Dt+divg(u)" must be converted to "(del_rho_t+dot(v,grad(rho)))+divg(u)" but "∂rho/∂t+divg(u)" must be converted to "del_rho_t+divg(u)".


---
Important:
**Even if the notation is SymPy-compatible, it still must be replaced.**
Variable names used in the notations above are "x" and "y", but they can be any variable names depending on the input, except for the differentiation operator characters given below. Variable names can be more than one letter, such as "rho" or "phi".
**Input variable names must NEVER be changed. Keep the original variable names exactly as they appear.**
Example: dp/dt or diff(p,t) must be converted to d_p_t, not d_y_x or any other forms using any variables other than p and t.
Differentiation operator characters are: "d", "D", "del", "∂", and "partial". The "**{{real number}}" such as "**2" or "**3" is also a part of the differentiation operator characters and not part of variable names.
**Variable names are always case-sensitive. If the variable is capitalized, it must be capitalized in the response.**
For example, if dp/dt or diff(p,t) is given, it must be converted to d_p_t, not d_P_t or any other forms.
The differentiation operators "d" and "D" are different; they are also case-sensitive.
Do not remove or add any characters except as required by the replacement rules.

All the notations in the same square brackets are equivalent.
**Notations from different groups or not listed above are NOT equivalent.**

---

This is the response: {expr}  
Now convert it to the required expression.  
Output only the expression, without any additional text or explanation.  
Remove all spaces. If there are any spaces remaining, you must remove them.
    """
    llm_response = llm.invoke(prompt)
    return llm_response

def convert_integral(expr: str, params: Params) -> str:
    load_dotenv()
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    prompt = fr"""
Follow these steps carefully:
A response is provided below. Convert it into an expression under the following rules.

When the following notations are found in the response, they must be replaced with the notation after the right arrow (->) after the square brackets.
---
Single integrals:
- [∫fdx, int(f,x), integrate(f,x), Integral(f,x)] -> intg(f,x)
- [∮fdx, int(f,x,circular=True), integrate(f,x,circular=True), Integral(f,x,circular=True), oint(f,x)] -> ointg(f,x)
- [∫ₐᵇfdx, ∫_a^bfdx, int_a^bfdx, int(f,(x,a,b)), integrate(f,(x,a,b)), Integral(f,(x,a,b))] -> intg(f,(x,a,b))
The above is a definite integral case.
- [∫ₐfdx, ∫_a^fdx, int_a^fdx, int(f,(x,a)), integrate(f,(x,a)), Integral(f,(x,a))] -> intg(f,x)
Example: ∫_{{V_sys}} t dV must be converted to intg(t,V).
As in the above example, if there is a lower limit and not an upper limit, it must be converted to intg(t,x), ignoring the lower limit.
- [Integral(f,(x,a),circular=True), oint(f,(x,a)), int(f,(x,a),circular=True), ∮ₐfdx, ∮_{{a}}fdx] -> ointg(f,x)
For the first three notations in the above group, the first argument of the second argument of the functions (Integral, oint, int) is the integration variable.
Do not assume that the integration variable is always x.
It must be "ointg", not "ointment"!
Multiple integrals:
- [∫∫fdxdy, int(int(f,x),y), integrate(f,x,y), Integral(f,x,y)] -> intg(intg(f,x),y)
- [∫∫∫fdxdydz, int(int(int(f,x),y),z), integrate(f,x,y,z), Integral(f,x,y,z)] -> intg(intg(intg(f,x),y),z)
Note that if there is an extra set of parentheses around "x,y,z", this is a definite case so it must be converted to intg(f,(x,y,z)).
All the integrals may have limits like for single integrals, for example: int_a^b(int_c^d(int_e^f(g,x)),y),z) must be converted to intg(intg(intg(g,(x,e,f)),(y,c,d)),(z,a,b)).

For all integrals, all the variables succeeding the letter 'd' **and not 'd_'** is the integration variable (what the function is integrated with respect to).
Example: ∫fdx means f is integrated with respect to x, and ∫∫fdxdy means f is integrated with respect to x and then y. Therefore the outputs must be intg(f,x) and intg(intg(f,x),y) respectively.
The integrand f can be any expression, including a function, a variable, or a constant, and it can be very long.
Example: ∫_{{V_sys}} del_rho_t+2 dV must be converted to intg(del_rho_t+2,V).

---
Important:
**Even if the notation is SymPy-compatible, it still must be replaced.**
Variable names used in the notations above are "f", "x", "y", "a" and "b", but they can be any variable names depending on the input, except for the differentiation operator characters given below. Variable names can be more than one letter, such as "rho" or "phi".
**Input variable names must NEVER be changed. Keep the original variable names exactly as they appear.**
Example: ∫pdt or int(p,t) must be converted to intg(p,t), not intg(y,x) or any other forms using any variables other than p and t.
**Variable names are always case-sensitive. If the variable is capitalized, it must be capitalized in the response.**
For example, if ∫pdt or int(p,t) is given, it must be converted to intg(p,t), not intg(P,t) or any other forms.
Do not remove or add any characters except as required by the replacement rules.

All the notations in the same square brackets are equivalent.
**Notations from different groups or not listed above are NOT equivalent.**

---

This is the response: {expr}  
Now convert it to the required expression.  
Output only the expression, without any additional text or explanation.  
Remove all spaces. If there are any spaces remaining, you must remove them.
    """
    llm_response = llm.invoke(prompt)
    return llm_response

def convert_other(expr: str, params: Params) -> str:
    load_dotenv()
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    prompt = fr"""
Follow these steps carefully:
A response is provided below. Convert it into an expression under the following rules.

When the following notations are found in the response, they must be replaced with the notation after the right arrow (->) after the square brackets.
---

- [∇f, gradient(f), grad(f)] -> grad(f)
- [∇·f, div(f), divergence(f)] -> divg(f)
Example: div(u) should be converted into divg(u).
- [∇×f, curl(f), rot(f)] -> rot(f)

- [Infinity, infinity, ∞, oo, Inf, inf, Infty, infty] -> oo
- [a·b, a⋅b, a.b, dot(a, b), a.dot(b)] -> dot(a,b)
  *Note: a.b is only equivalent to dot(a,b) if a and b are variables, not constants like 0, 1, π, etc.*
- [a×b, cross(a, b), a.cross(b)] -> cross(a,b)
- [\vec{{a}}, vector(a), a.vector(), Matrix(a)] -> vec(a)
- [â, \hat{{a}}, unit(a), normalize(a), a_hat] -> hat(a)
- [exp(x), e**x, e**x, exponential(x)] -> exp(x)
These can be within an integral, for example:
"Integral(rho * u.dot(n), (A, A_sys), circular=True)" must be converted into "Integral(rho * dot(u,n), (A, A_sys), circular=True)".

---
Important:
**Even if the notation is SymPy-compatible, it still must be replaced.**
Variable names used in the notations above are "f", "x", "a" and "b", but they can be any variable names depending on the input, except for the differentiation operator characters given below. Variable names can be more than one letter, such as "rho" or "phi".
**Input variable names must NEVER be changed. Keep the original variable names exactly as they appear.**
Example: curl(p) must be converted to rot(p), not rot(f) or any other forms using any variables other than p.
**Variable names are always case-sensitive. If the variable is capitalized, it must be capitalized in the response.**
Example: curl(p) must be converted to rot(p), not rot(P) or any other forms.
Do not remove or add any characters except as required by the replacement rules.

All the notations in the same square brackets are equivalent.
**Notations from different groups or not listed above are NOT equivalent.**

---

This is the response: {expr}  
Now convert it to the required expression.  
Output only the expression, without any additional text or explanation.  
Remove all spaces. If there are any spaces remaining, you must remove them.
    """
    llm_response = llm.invoke(prompt)
    return llm_response