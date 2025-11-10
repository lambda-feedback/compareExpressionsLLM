import re
from typing import Any, TypedDict
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def parse_markdown_with_images(markdown_text: str):
    """Parse markdown text that may contain embedded images (![](url))"""
    pattern = r'!\[.*?\]\((.*?)\)'
    content = []
    last_end = 0

    for match in re.finditer(pattern, markdown_text):
        start, end = match.span()
        url = match.group(1).strip()

        text_before = markdown_text[last_end:start].strip()
        if text_before:
            content.append({"type": "text", "text": text_before})

        content.append({"type": "image_url", "image_url": {"url": url}})
        last_end = end

    remaining = markdown_text[last_end:].strip()
    if remaining:
        content.append({"type": "text", "text": remaining})
    return content


def eval_with_feedback(
    question_markdown: str,
    part_markdown: str,
    pre_response_text: str,
    student_answer: str,
    post_response_text: str,
    correct_answer: str,
) -> str:
    """
    Evaluate a student's answer based on combined context from:
    - Question (text + images)
    - Part (text + images)
    - Pre and post response text
    """
    load_dotenv()

    llm = ChatOpenAI(
        model=os.environ["OPENAI_MODEL"],  # must support image input (e.g. gpt-4o, gpt-5)
        api_key=os.environ["OPENAI_API_KEY"],
    )

    # Parse both question and part markdowns
    question_content = parse_markdown_with_images(question_markdown)
    part_content = parse_markdown_with_images(part_markdown)

    # Feedback generation instruction prompt
    instruction_text = fr"""
Follow these steps carefully:

You are given:
- A question and its sub-part (each may include diagrams or equations).
- The pre-response text and post-response text that appear around the student's answer box.
- The student's answer and the correct answer.

Your task:
1. Understand the problem statement and its context (including the question, part, and images).
2. Analyze the reasoning that leads from the question to the correct answer.
3. Identify *why* the student’s answer might differ (conceptual misunderstanding, skipped step, sign/unit error, etc.).
4. Write one **short, indirect feedback sentence** that:
   - Encourages the student to rethink that specific step or concept (thought trigger), and
   - Refers to the relevant mathematical action or context (action trigger).
5. Do NOT reveal the correct formula or result.

Guidelines:
- Use imperative mood: "Re-examine...", "Review...", "Reconsider...", "Verify...".
- Mention a specific step or operation, e.g. "when integrating", "when substituting", "when solving for x".
- Keep it concise (max 15 words).
- Be constructive and professional.

Now, generate only the final feedback sentence.

Pre-response text: {pre_response_text}
Student's answer (LaTeX): {student_answer}
Post-response text: {post_response_text}
Correct answer (LaTeX): {correct_answer}

Output only the feedback sentence.
"""

    # Combine all content, preserving order and image placement
    full_content = (
        [{"type": "text", "text": "Main question:"}]
        + question_content
        + [{"type": "text", "text": "\nSub-part:"}]
        + part_content
        + [{"type": "text", "text": instruction_text}]
    )

    response = llm.invoke([{"role": "user", "content": full_content}])
    return response.content.strip()
