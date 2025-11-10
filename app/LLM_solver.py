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

def LLM_solve(question_txt: str, input_type: str, pre_response_txt: str, post_response_txt: str) -> str:
    load_dotenv()
    llm = ChatOpenAI(
        model=os.environ['OPENAI_MODEL'],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    prompt = fr"""
    Follow these steps carefully:

    A question text and its "input type" are given at the end of this prompt.
    The task is to answer the question only, without any additional explanation.
    The question is in the topic of either mathematics or science.
    The "input type" can be one of the following:
    BOOLEAN, EXPRESSION, MATRIX, MULTIPLE_CHOICE, NUMBER, NUMERIC_UNITS, TEXT.

    For BOOLEAN type, answer either "True" or "False" only.
    For EXPRESSION type, answer with a mathematical expression in LaTeX format.
    For MATRIX type, answer with in LaTeX format the expressions or numbers for each of the element, from left to right and from top to bottom.
    For MULTIPLE_CHOICE type, answer with either 1st, 2nd, 3rd, or 4th only.
    For NUMBER type, answer with a number only.
    For NUMERIC_UNITS type, answer with a number followed by a space and the unit, e.g., "9.8 m/s^2".
    For TEXT type, answer with a short text  without any explanation. The answer is usually 2 words or less.

    The "Pre response text" and "Post response text" are also given to you at the end of this prompt to help you understand the context.
    For example, if the answer is "x=yz" and "Pre response text" is "x=", then you should answer with "yz" only.

    Question text:
    {question_txt}
    Input type:
    {input_type}
    Pre response text:
    {pre_response_txt}
    Post response text:
    {post_response_txt}

    Now answer the question.
    """

    expr = llm.invoke(prompt).content.strip()

    return expr
