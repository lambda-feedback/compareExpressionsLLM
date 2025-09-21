from typing import Any, TypedDict
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import re
import requests
import base64


class Params(TypedDict):
    pass


class Result(TypedDict):
    preview: Any

def input_symbol_conversion(response: str, mapping: dict) -> str:
    """
    Convert internal symbols in response to LaTeX equivalents,
    and wrap the entire expression in $...$ (no nested $).
    
    mapping: {latex: internal}  e.g. {r"\kappa": "kappa", r"\rho": "rho", "U_0": "U0"}
    """
    reverse_map = {}
    for latex_sym, internal_sym in mapping.items():
        # "$\kappa$" → "\kappa"
        clean_latex = latex_sym.strip("$")
        reverse_map[internal_sym] = clean_latex

    converted = response
    for internal_sym, latex_sym in reverse_map.items():
        pattern = r'\b{}\b'.format(re.escape(internal_sym))
        converted = re.sub(pattern, lambda m: latex_sym, converted)

    return f"{converted}"

# mapping = {
#     r"$\kappa$": "kappa",
#     r"$\rho$": "rho",
#     "U_0": "U0"
# }

# response = "(kappa+1)/rho + U0"
# print(imput_symbol_conversion(response, mapping))

#"Draw" mode or "Scan" mode
def mathpix_to_latex(image_path):
    load_dotenv()
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    r = requests.post("https://api.mathpix.com/v3/text",
        headers={"app_id": os.environ['MATHPIX_APP_ID'], "app_key": os.environ['MATHPIX_APP_KEY']},
        json={
            "src": f"data:image/jpg;base64,{image_data}",
            "formats": ["latex"]
        }
    )
    return r.json()["latex_normalized"]

#"Type" mode
def preview_function(response: str, mapping: dict) -> str:
    load_dotenv()
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    response = input_symbol_conversion(response, mapping)

    prompt_latex = fr"""
    Follow these steps carefully:

    1. Input the student response given right after these steps.
        - If the response is empty, output an empty string and nothing else.
        - Otherwise, the response may be in LaTeX format, sympy format or a mixture of both. Follow the steps below to convert it into LaTeX format.

    2(a). If the response is in LaTeX format, output it exactly as given without adding or modifying any backslashes.
    2(b). Convert the student response into LaTeX form. 
    Do not wrap the expression with `\(` at the start and `\)` at the end of it. Do not line break.  
    ✅ Use only single backslashes in the LaTeX commands.  
    ⚠ Before converting or outputting, strip any invisible or control characters such as \x0c, \x0b, \r, etc. Remove them entirely from the response.
    ⚠ Never escape backslashes (no `\\` or `\\\\`).  
    ⚠ Return only the raw LaTeX code, exactly as it would appear in a LaTeX document. Do **not** format it as a Python string or JSON string.
    Use `\left(` and `\right)` to wrap parentheses, including when they are used for functions like `sin`, `cos`, `tan`, `ln`, etc. 
    For functions like `sin`, `cos`, `tan`, `ln`, etc, use `\sin`, `\cos`, `\tan`, `\ln`, etc., i.e. always add a backslash before the function name.
    Do not calculate or simplify the expression. eg. "2+2" should be "2+2", not "4".

    3. Output the LaTeX expression only. Do not include any explanation, quotes, or escaping.

    This is the student response: {response}
    Now convert it to LaTeX format as per the above instructions.
    """

    expr = llm.invoke(prompt_latex).content.strip()

    expr = re.sub(
        r'([A-Za-z0-9_]+|\))\^(\d+|[A-Za-z]|\([^()]+\))',
        lambda m: f"{m.group(1)}^{{{m.group(2)}}}",
        expr
    )
    expr = re.sub(
        r'([A-Za-z0-9_]+|\))\_(\d+|[A-Za-z]|\([^()]+\))',
        lambda m: f"{m.group(1)}_{{{m.group(2)}}}",
        expr
    )

    expr = re.sub(
        r'\\(sin|cos|tan|log|ln|exp)\((.*?)\)',
        lambda m: f" \\{m.group(1)}\\left({m.group(2)}\\right)",
        expr
    )
    expr = expr.replace(r'\left(', r' \left(')
    expr = expr.replace(r'\right)', r' \right)')
    expr = expr.replace('*', ' ')

    return f"${expr}$"
