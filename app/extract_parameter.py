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

def extract_parameter(question_txt: str) -> str:
    load_dotenv()
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    prompt = fr"""
    Follow these steps carefully:

    A question text is given at the end of this prompt. 
    The task is to extract the mathematical conditions from the question text and put them in the lists.
    The question is in the topic of either mathematics or science.
    The conditions are mainly classified into the following two types:
    1) Conditions that define the properties of a constant (e.g., "x is a real number", "y is a complex number", "x > 0", etc.)
    2) Conditions that define the types of variables (e.g., "y is a function of x", "f is a matrix", "u is a vector", etc.)
    These may be combined together (e.g., "y is a real-valued function of x", "A is a positive definite matrix", etc.)
    
    The output format is as follows:
    For type 1) conditions, output in the format:
    "symbol_assumptions"={{
        "<symbol>": {{"<property>": <value>, ...}}, ...}}
    where <property> can be:
        - "real": True/False
        - "integer": True/False
        - "positive": True/False
        - "negative": True/False
        - "complex": True/False
        - "nonzero": True/False
    Example:
        "x is a real number" → {{"x": {{"real": True}}}}
        "n is a positive integer" → {{"n": {{"integer": True, "positive": True}}}}
    If a variable is mentioned in the question but no specific property is given, do not include it in the "symbol_assumptions" dictionary.
    Example:
        "u is a function of x and y" → no entry for "u" in "symbol_assumptions"

    For type 2) conditions, output in the format:
    "function"=["f(x)", "y(x,z)", ...]
    Example:
        "y is a function of x" → ["y(x)"]
        "f is a function of x and z" → ["f(x,z)"]

    If both types are present, include both in the output dictionary.

    Return the result strictly as a JSON-like Python dictionary with keys:
    {{
        "symbol_assumptions"={{...}},
        "function"=[...]
    }}

    Do not include explanations, output only the dictionary.

    Question text:
    {question_txt}
    Now output the dictionary.
    """

    expr = llm.invoke(prompt).content.strip()

    return expr
