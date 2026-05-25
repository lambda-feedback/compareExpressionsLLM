# compareExpressionsLLM

An AWS Lambda evaluation function for the [Lambda Feedback](https://lambda-feedback.github.io/Documentation/) platform that checks whether a student's mathematical expression is equivalent to the correct answer.

Equivalence is determined primarily by symbolic computation via **SymPy**, with optional LLM-powered helpers (OpenAI) for parameter extraction, answer generation, and student feedback.

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
  - [Evaluation](#evaluation)
  - [Preview](#preview)
  - [LLM Utilities](#llm-utilities)
- [Setup](#setup)
  - [Environment Variables](#environment-variables)
  - [Getting Started](#getting-started)
- [Deployment](#deployment)
- [Contact](#contact)

## Features

- **Symbolic equivalence** — uses SymPy to simplify and compare expressions, equations, and matrices
- **Special math syntax** — handles differentials, integrals, summations, and vector calculus operators (`∇`, `∮`, `∂`, etc.) via regex-based conversion
- **Fluid mechanics domain** (`FMX2_symbols`) — pre-loaded symbols and operators for velocity, pressure, temperature, gradients, divergence, and curl
- **Symbol assumptions** — supports `real`, `positive`, `integer`, `complex`, `nonzero` constraints on variables
- **Function declarations** — treats named symbols as functions of specified variables (e.g. `y(x)`)
- **Domain checking** — validates that numeric answers lie within a specified interval
- **Greek symbol support** — Unicode Greek letters (`α`, `β`, …) are normalised to ASCII names before parsing
- **LLM parameter extraction** — extracts mathematical conditions from question text (symbol assumptions, function declarations, domains)
- **LLM question solving** — generates an expected answer given a question, input type, and context
- **LLM feedback generation** — produces a short, constructive hint when the student's answer is wrong
- **LaTeX preview** — converts student input to rendered LaTeX (typed input via LLM; handwritten/scanned input via Mathpix OCR)

## Repository Structure

```
app/
    evaluation.py             # Main evaluation_function entry point
    evaluation_symbolise.py   # Operator-based symbolisation fallback
    preview.py                # preview_function + Mathpix OCR helper
    LLM_solver.py             # LLM-based question answering
    evaluate_with_feedback.py # LLM-based student feedback generation
    extract_parameter.py      # LLM-based parameter extraction from question text
    parameter.py              # SymPy symbol/function parsing utilities
    re_conversion.py          # Regex conversion of special math syntax
    llm_conversion.py         # LLM conversion of special math syntax
    FMX2_symbols.py           # Fluid mechanics symbol definitions (FMX2 module)
    evaluation_tests.py       # Unit tests for evaluation_function
    evaluation_test_cases.py  # Test case data for evaluation
    preview_tests.py          # Unit tests for preview_function
    preview_test_cases.py     # Test case data for preview
    LLM_solver_testing.py     # Tests for LLM solver
    evaluate_with_feedback_testing.py  # Tests for feedback generation
    extract_parameter_tests.py         # Tests for parameter extraction
    miscellaneous_testing.py           # Miscellaneous experiments
    requirements.txt          # Python dependencies

.github/
    workflows/
        test-and-deploy.yml   # CI/CD pipeline

config.json   # Evaluation function name for deployment
.gitignore
```

## How It Works

### Evaluation

`evaluation_function(response, answer, params)` in `evaluation.py`:

1. **Pre-processing** — rejects unbalanced parentheses or expressions starting/ending with a bare operator; normalises `^` to `**`; replaces Unicode Greek letters
2. **Special syntax detection** — checks whether either expression contains differentials, integrals, or other advanced operators
3. **Conversion** — if special syntax is detected, regex rules in `re_conversion.py` convert to SymPy-compatible form (LLM conversion in `llm_conversion.py` is available as an alternative)
4. **SymPy comparison** (`is_equivalent_sympy`) — parses both expressions with `parse_expr`, applies any declared symbol assumptions and function declarations, then checks `simplify(expr1 - expr2) == 0`; for equations, solves both and compares solution sets; for matrices, checks the zero matrix
5. **Domain check** — if a domain is specified in `params`, validates numeric answers against the interval
6. **Fallback** — if SymPy cannot solve, `symbolise_by_operators` maps sub-expressions to placeholder symbols and retries

`params` keys:

| Key | Type | Description |
| --- | --- | --- |
| `symbol_assumptions` | `dict` | Per-symbol SymPy assumptions, e.g. `{"x": {"real": true}}` |
| `function` | `list[str]` | Declared function specs, e.g. `["y(x)", "f(x,z)"]` |
| `domain` | `str` | Interval string, e.g. `"(0,1)"`, `"[-1,1]"` |

### Preview

`preview_function(response, mapping)` in `preview.py`:

- Applies a symbol mapping (internal names → LaTeX) to the raw student input
- Calls an OpenAI LLM to convert the (possibly mixed) input into clean LaTeX
- Applies post-processing regex fixes for exponents, subscripts, and function formatting
- Returns the result wrapped in `$...$`

For handwritten or scanned input, `mathpix_to_latex(image_path)` calls the Mathpix API.

### LLM Utilities

| Module | Function | Purpose |
| --- | --- | --- |
| `LLM_solver.py` | `LLM_solve(question, input_type, pre, post)` | Generates the expected answer for a question |
| `evaluate_with_feedback.py` | `eval_with_feedback(question_md, part_md, pre, student_answer, post, correct_answer)` | Returns a short pedagogical hint (≤ 15 words) for an incorrect answer; supports embedded images in markdown |
| `extract_parameter.py` | `extract_parameter(question_txt)` | Extracts symbol assumptions, function declarations, and domain from question text |

## Setup

### Environment Variables

Create a `.env` file (or set these in your environment):

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o  # any model that supports vision for feedback generation

# Required only for Mathpix OCR (handwritten input)
MATHPIX_APP_ID=...
MATHPIX_APP_KEY=...
```

### Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```
3. Copy `.env.example` to `.env` (if provided) and fill in your API keys
4. Set the function name in `config.json`:
   ```json
   { "EvaluationFunctionName": "compareExpressionsLLM" }
   ```
5. Run the tests:
   ```bash
   python -m pytest app/
   ```

## Deployment

Merging into the default branch triggers `.github/workflows/test-and-deploy.yml`, which:

1. Runs all unit tests
2. Builds a Docker image of the app
3. Pushes it to the shared Lambda Feedback ECR repository
4. Calls the backend `grading-function/ensure` route to deploy the function

For more information on the Lambda Feedback infrastructure, see the [BaseEvaluationFunctionLayer](https://github.com/lambda-feedback/BaseEvalutionFunctionLayer) repository.

## Contact

For questions about this function, contact the repository maintainer.
For platform issues, file a report on the [Lambda Feedback Documentation](https://lambda-feedback.github.io/Documentation/) site.