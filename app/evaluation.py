import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from typing import Any, TypedDict
from sympy import solve, Eq, simplify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re


class Params(TypedDict):
    pass


class Result(TypedDict):
    is_correct: bool
    sympy_result: bool | None
    llm_result: bool
    mismatch_info: str


transformations = standard_transformations + (implicit_multiplication_application,)


def contains_special_math(expr: str) -> bool:
    """
    特殊な記号/演算が含まれているか判定
    """
    patterns = [
        r"d(\^|\*\*)?\d*(\*\*)?\w*/d\w+(\^|\*\*)?\d*(\*\*)?", # Ordinary diff (dy/dx, d^2y/dx^2)
        r"∂(\^|\*\*)?\d*(\*\*)?\w*/∂\w+(\^|\*\*)?\d*(\*\*)?",        # Partial diff (∂y/∂x, ∂^2y/∂x^2)
        r"diff\(\w+, \w+\)",                # diff function (diff(y, x))
        r"int",                          # integration (int_b^a f(x)dx)
        r"∫",                            
    ]
    return any(re.search(p, expr) for p in patterns)


def is_equivalent_sympy(expr1, expr2) -> bool | None:
    """
    Return True/False if comparable with SymPy,
    or None if an error occurs.
    """
    try:
        expr1, expr2 = expr1.replace("^", "**"), expr2.replace("^", "**")
        if not expr1.strip() and not expr2.strip():
            return True
        elif not expr1.strip() or not expr2.strip():
            return False

        # Compare with Eq() for equations
        if "=" in expr1 and "=" in expr2:
            lhs1, rhs1 = expr1.split("=")
            lhs2, rhs2 = expr2.split("=")

            # implicit multiplication handlable
            lhs1_parsed = parse_expr(lhs1, transformations=transformations)
            rhs1_parsed = parse_expr(rhs1, transformations=transformations)
            lhs2_parsed = parse_expr(lhs2, transformations=transformations)
            rhs2_parsed = parse_expr(rhs2, transformations=transformations)

            eq1 = Eq(lhs1_parsed - rhs1_parsed, 0)
            eq2 = Eq(lhs2_parsed - rhs2_parsed, 0)

            all_symbols = eq1.free_symbols.union(eq2.free_symbols)

            sol1 = solve(eq1, list(all_symbols))
            sol2 = solve(eq2, list(all_symbols))

            return set(sol1) == set(sol2)
        else:
            expr1_parsed = parse_expr(expr1, transformations=transformations)
            expr2_parsed = parse_expr(expr2, transformations=transformations)
            return simplify(expr1_parsed - expr2_parsed) == 0

    except Exception as e:
        print(f" SymPy error: {e}")
        return None


def evaluation_function(response, answer, params):
    load_dotenv()
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # Check if LLM priority is needed
    needs_llm_priority = contains_special_math(response) or contains_special_math(answer)

    # Check with SymPy first if not using LLM priority
    sympy_result = None
    if not needs_llm_priority:
        sympy_result = is_equivalent_sympy(response, answer)

    prompt = fr"""
    Follow these steps carefully:
    A student response and an answer are provided below. Compare the two if they are mathematically equivalent.
    Only return True if they are **exactly equivalent** for all possible values of all variables.
    Do not assume expressions are equivalent based on similarity.
    There are a few types of symbols for differentiation and the following in the same square brackets are considered equivalent:
    [dy/dx, d/dx(y), diff(y,x)], [d^2y/dx^2, d**2y/dx**2, diff(y,x,x)], [∂y/∂x, ∂/∂x(y), diff(y,x), partial(y)/partial(x)], [∂^2y/∂x^2, ∂**2y/∂x**2, diff(y,x,x), partial**2(y)/partial(x)**2, partial^2(y)/partial(x)^2]
    The terms above that are not in the same square brackets are not considered equivalent.
    Student response: {response}
    Answer: {answer}

    Return either True or False as a single word and nothing else.
    """
    llm_response = llm.invoke(prompt)
    llm_result_text = llm_response.content.strip().lower()

    if llm_result_text == "true":
        llm_result = True
    elif llm_result_text == "false":
        llm_result = False
    else:
        # Any weird responses
        llm_result = False

    if sympy_result is not None:
        if sympy_result == llm_result:
            return {
                "is_correct": sympy_result,
                "sympy_result": sympy_result,
                "llm_result": llm_result,
                "mismatch_info": ""
            }
        else:
            mismatch_info = (
                f"Mismatch detected:\n"
                f"- SymPy result: {sympy_result}\n"
                f"- LLM result: {llm_result}\n"
                f"Used LLM result due to mismatch"
            )
            return {
                "is_correct": sympy_result, 
                "sympy_result": sympy_result,
                "llm_result": llm_result,
                "mismatch_info": mismatch_info
            }
    else:
        return {
            "is_correct": llm_result,
            "sympy_result": None,
            "llm_result": llm_result,
            "mismatch_info": "Used LLM result only"
        }