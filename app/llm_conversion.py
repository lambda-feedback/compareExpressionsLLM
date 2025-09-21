import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from parameter import Params
load_dotenv()

def convert_diff(expr: str, params: Params) -> str:
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
                )
    prompt = fr"""
Follow these steps carefully:
A response is provided below. Convert it into an expression under the following rules.

When the following notations are found in the response, they must be replaced with the notation after the right arrow (->) after the square brackets.

---

- [dy/dx, d/dx(y), diff(y,x)] -> smart_derivative(y,x,1)

- [d^2y/dx^2, d**2y/dx**2, diff(y,x,x)] -> smart_derivative(y,x,2)
- [d^3y/dx^3, d**3y/dx**3, diff(y,x,x,x)] -> smart_derivative(y,x,3)
- [∂y/∂x, ∂/∂x(y), partial(y)/partial(x), dely/delx, del(y)/del(x)] -> smart_derivative(y,x,1)
- [∂**2y/dx**2, diff(y,x,x), partial**2(y)/partial(x)**2] -> smart_derivative(y,x,2)
- [Dy/Dt, D/Dt(y)] -> (smart_derivative(y,t,1)+smart_dot(Gradient(y),u_vec))
Do NOT convert to (smart_derivative(y,t,1)+Gradient(y).dot(u_vec)).
Do NOT convert to (smart_derivative(y,t,1)+Gradient(y)*u_vec).
Never omit the second term, "+smart_dot(Gradient(y),u_vec)".
Do NOT remove any surrounding terms like "+Divergence(...)" when replacing "Drho/Dt" or "D/Dt(...)".  
If the expression is "Drho/Dt + Divergence(u_vec) = 0", the correct output is "(smart_derivative(rho,t,1)+smart_dot(Gradient(rho),u_vec))+Divergence(u_vec)=0".  
Never omit "Divergence(u_vec)".
---
Important:
**Even if the notation is SymPy-compatible, it still must be replaced.**
Variable names used in the notations above are "x" and "y", but they can be any variable names depending on the input, except for the differentiation operator characters given below. Variable names can be more than one letter, such as "rho" or "phi".
**Input variable names must NEVER be changed. Keep the original variable names exactly as they appear.**
Example: dp/dt or d/dt(p) must be converted to smart_derivative(p,t,1), not smart_derivative(y,x,1) or any other forms using any variables other than p and t.
Differentiation operator characters are: "d", "D", "del", "∂", and "partial". The "**{{real number}}" such as "**2" or "**3" is also a part of the differentiation operator characters and not part of variable names.
**Variable names are always case-sensitive. If the variable is capitalized, it must be capitalized in the response.**
For example, if dp/dt or d/dt(p) is given, it must be converted to smart_derivative(p,t,1), not smart_derivative(P,t,1) or any other forms.
The differentiation operators "d" and "D" are different; they are also case-sensitive.
Example: if the input is "DT/Dt", you must convert this to "(smart_derivative(T,t,1)+Gradient(T)*u_vec)".
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
- [∫fdx, int(f,x), integrate(f,x), Integral(f,x)] -> Integral(f,x)
- [∮fdx, int(f,x,circular=True), integrate(f,x,circular=True), Integral(f,x,circular=True), oint(f,x)] -> Integral(f,(x,0,2*pi))
- [∫ₐᵇfdx, ∫_a^bfdx, int_a^bfdx, int(f,(x,a,b)), integrate(f,(x,a,b)), Integral(f,(x,a,b))] -> Integral(f,(x,a,b))
The above is a definite integral case.
- [∫ₐfdx, ∫_a^fdx, int_a^fdx, int(f,(x,a)), integrate(f,(x,a)), Integral(f,(x,a))] -> Integral(f,x)
Example: ∫_{{V_sys}} t dV must be converted to Integral(t,V).
As in the above example, if there is a lower limit and not an upper limit, it must be converted to Integral(t,x), ignoring the lower limit.
- [Integral(f,(x,a),circular=True), oint(f,(x,a)), int(f,(x,a),circular=True), ∮ₐfdx, ∮_{{a}}fdx] -> Integral(f,(x,0,2*pi))
For the first three notations in the above group, the first argument of the second argument of the functions (Integral, oint, int) is the integration variable.
Do not assume that the integration variable is always x.
Multiple integrals:
- [∫∫fdxdy, int(int(f,x),y), integrate(f,x,y), Integral(f,x,y)] -> Integral(Integral(f,x),y)
- [∫∫∫fdxdydz, int(int(int(f,x),y),z), integrate(f,x,y,z), Integral(f,x,y,z)] -> Integral(Integral(Integral(f,x),y),z)
All the integrals may have limits like for single integrals, for example: int_a^b(int_c^d(int_e^f(g,x)),y),z) must be converted to Integral(Integral(Integral(g,(x,e,f)),(y,c,d)),(z,a,b)).

The integrand f can be any expression, including a function, a variable, or a constant, and it can be very long.
Example: ∫_{{V_sys}} del_rho_t+2 dV must be converted to Integral(del_rho_t+2,V).

---
Important:
**Even if the notation is SymPy-compatible, it still must be replaced.**
Variable names used in the notations above are "f", "x", "y", "a" and "b", but they can be any variable names depending on the input, except for the differentiation operator characters given below. Variable names can be more than one letter, such as "rho" or "phi".
**Input variable names must NEVER be changed. Keep the original variable names exactly as they appear.**
Example: ∫pdt or int(p,t) must be converted to Integral(p,t), not Integral(y,x) or any other forms using any variables other than p and t.
**Variable names are always case-sensitive. If the variable is capitalized, it must be capitalized in the response.**
For example, if ∫pdt or int(p,t) is given, it must be converted to Integral(p,t), not Integral(P,t) or any other forms.
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
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
       )
    prompt = fr"""
Follow these steps carefully:
A response is provided below. Convert it into an expression under the following rules.

When the following notations are found in the response, they must be replaced with the notation after the right arrow (->) after the square brackets.
---

- [∇f, gradient(f), grad(f), nabla(f)] -> Gradient(f)
- [∇·f, div(f), divergence(f), nabla·f] -> Divergence(f)
Example: div(u) should be converted into Divergence(u).
- [∇×f, curl(f), rot(f), nablaxf, ∇xf] -> Curl(f)

- [Infinity, infinity, ∞, oo, Inf, Infty, infty] -> inf
- [a·b, a⋅b, a.b, dot(a,b)] -> smart_dot(a,b)

Example: "grad(u_vec)*u_vec" must be converted to "Gradient(u_vec)*u_vec" not "Gradient(u_vec).dot(u_vec)".
  *Note: a.b is only equivalent to a.dot(b) if a and b are variables, not constants like 0, 1, π, etc.*
  *Note: a and b are not commutative. Therefore, when for example "u_vec·x_vec" is input, you must convert it into "u_vec.dot(x_vec)".
- [a×b, cross(a,b)] -> a.cross(b)
- [\vec{{a}}, vector(a), a.vector(), Matrix(a)] -> Matrix(a)
- [â, \hat{{a}}, unit(a), normalize(a), hat(a)] -> a_hat
- [exp(x), e**x, e**x, exponential(x)] -> exp(x)
These can be within an integral, for example:
"Integral(rho * u.dot(n), (A, A_sys), circular=True)" must be converted into "Integral(rho * dot(u,n), (A, A_sys), circular=True)".

---
Important:
**Even if the notation is SymPy-compatible, it still must be replaced.**
Variable names used in the notations above are "f", "x", "a" and "b", but they can be any variable names depending on the input, except for the differentiation operator characters given below. Variable names can be more than one letter, such as "rho" or "phi".
Example: "hat(r)" must be converted to "r_hat", not "a_hat".
**Input variable names must NEVER be changed. Keep the original variable names exactly as they appear.**
Example: curl(p) must be converted to Curl(p), not Curl(f) or any other forms using any variables other than p.
**Variable names are always case-sensitive. If the variable is capitalized, it must be capitalized in the response.**
Example: curl(p) must be converted to Curl(p), not Curl(P) or any other forms.
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