import asyncio
import multiprocessing
import os
import pathlib
import queue
import re
import subprocess
import sys
import time
import traceback
from typing import Any, List

import weave
import openai
import instructor

from pydantic import BaseModel, Field
from tree_sitter_languages import get_language, get_parser


# API params
BASE_URL = os.getenv("BASE_URL", None)
API_KEY = os.getenv("API_KEY", "dummy_key")

# params
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
FAST_LLM = os.getenv("FAST_LLM", "open-mistral-nemo-2407")
STRONG_LLM = os.getenv("STRONG_LLM", "mistral-large-latest")

# API client
oai_client = openai.AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)
async_client = instructor.from_openai(oai_client, mode=instructor.Mode.JSON)

language = get_language("python")
tree_parser = get_parser("python")


def remove_extra_newlines(text: str) -> str:
    # Use regex to replace 2 or more newlines (with possible whitespace in between) with a single newline
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text


def remove_comments_and_docstrings(code):
    # Define queries to capture comments and docstrings
    doc_str_pattern = """
    (module . (expression_statement (string)) @module_doc_str)
    (class_definition body: (block . (expression_statement (string)) @class_doc_str))
    (function_definition body: (block . (expression_statement (string)) @function_doc_str))
    """

    comment_pattern = "(comment) @comment"
    # Parse the code
    tree = tree_parser.parse(code.encode())
    root_node = tree.root_node

    # Query the tree for docstrings and comments
    doc_str_query = language.query(doc_str_pattern)
    doc_strs = doc_str_query.captures(root_node)

    comment_query = language.query(comment_pattern)
    comments = comment_query.captures(root_node)

    # Get the start and end points of all docstrings and comments
    doc_str_points = set((node.start_byte, node.end_byte) for node, _ in doc_strs)
    comment_points = set((node.start_byte, node.end_byte) for node, _ in comments)

    # Create a set of all points to remove
    remove_points = doc_str_points.union(comment_points)

    # Reconstruct the code, skipping over the parts to remove
    cleaned_code = []
    last_index = 0
    for start, end in sorted(remove_points):
        if last_index < start:
            cleaned_code.append(code[last_index:start])
        last_index = end

    # Add any remaining code after the last comment/docstring
    cleaned_code.append(code[last_index:])

    return "".join(cleaned_code)


def clean_code_string(code: str) -> str:
    code = remove_comments_and_docstrings(code)
    code = remove_extra_newlines(code)
    return code

class TestReport(BaseModel):
    status: str
    message: str

    @property
    def as_xml(self) -> str:
        return f"""
<test_report>
<status>{self.status}</status>
<message>{self.message}</message>
</test_report>
"""

async def exec_program(program, input_data, expected_output, timeout):
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", program,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(input=input_data.encode()), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            return TestReport(
                status="timeout",
                message=f"Took too long! Your program timed out after {timeout} seconds of execution."
            )
        
        if process.returncode != 0:
            return TestReport(
                status="error", message=f"Program execution failed: {stderr.decode()}"
            )
        else:
            if stdout.decode().strip() == expected_output.strip():
                return TestReport(
                    status="passed", message="Yay! Your program ran successfully"
                )
            else:
                return TestReport(
                    status="failed",
                    message=f"<expected>\n{expected_output}</expected>\n---\n<got>\n{stdout.decode()}</got>",
                )
    except Exception:
        return TestReport(
            status="error", message=f"An error occurred: {traceback.format_exc()}"
        )

@weave.op
async def check_correctness(
    program: str, input_data: str, expected_output: str, timeout: float
) -> TestReport:
    return await exec_program(program, input_data, expected_output, timeout)

import re

def sanitize_unicode(json_data):
    # Use a regular expression to remove control characters
    sanitized_data = re.sub(r'[\u0000-\u001F]+', '', json_data)
    return sanitized_data


@weave.op
async def format_response(text: str, model: Any, temperature: float = 0.1) -> Any:
    text = sanitize_unicode(text)
    formatted_response = await async_client.chat.completions.create(
        model=FAST_LLM,
        # Instructor adds a system message by default about how to format the response given the response model.
        messages=[
            {
                "role": "user",
                "content": f"Extract the relevant information from the following document and return it in valid JSON\n\n{text}",
            }
        ],
        temperature=temperature,
        response_model=model,
        max_retries=2,
        max_tokens=MAX_TOKENS,
    )
    return formatted_response


class Problem(BaseModel):
    problem_dir: pathlib.Path = Field(
        ..., description="The path to the problem directory"
    )
    problem_name: str = Field(..., description="The name of the problem")
    problem_description: str = Field(..., description="The description of the problem")
    sample_input: str = Field(..., description="The sample input of the problem")
    sample_output: str = Field(..., description="The sample output of the problem")
    problem_input: pathlib.Path = Field(..., description="The path to the input file")
    problem_output: pathlib.Path = Field(..., description="The path to the output file")

    @property
    def as_xml(self) -> str:
        return f"""
<problem>
<problem_statement>
{remove_extra_newlines(self.problem_description)}
</problem_statement>
<sample_test_cases>
<sample_input>
{self.sample_input}
</sample_input>
<sample_output>
{self.sample_output}
</sample_output>
</sample_test_cases>
</problem>
"""


def load_problem(problem_name: str, problem_dir: pathlib.Path) -> Problem:
    problem_input = problem_dir / f"{problem_name}.in"
    problem_output = problem_dir / f"{problem_name}.out"
    sample_input = problem_dir / f"{problem_name}_sample_input.txt"
    sample_output = problem_dir / f"{problem_name}_sample_output.txt"
    problem_description = problem_dir / f"{problem_name}.md"
    return Problem(
        problem_dir=problem_dir,
        problem_name=problem_name,
        problem_description=problem_description.read_text(),
        sample_input=sample_input.read_text(),
        sample_output=sample_output.read_text(),
        problem_input=problem_input,
        problem_output=problem_output,
    )


class Analysis(BaseModel):
    core_question: str = Field(..., description="Core question of the problem")
    problem_solving_info: List[str] = Field(
        ..., description="Problem-solving information related to the core question"
    )
    algorithm: str = Field(..., description="Algorithm to solve the problem")
    tutorial: str = Field(..., description="Tutorial on the algorithm")
    plan: str = Field(..., description="Step by step plan to solve the problem")
    pseudocode: str = Field(..., description="Pseudocode to solve the problem")

    @property
    def as_xml(self) -> str:
        return f"""
<core_question>
{self.core_question}
</core_question>
<problem_solving_info>
{self.problem_solving_info}
</problem_solving_info>
<algorithm>
{self.algorithm}
</algorithm>
<tutorial>
{self.tutorial}
</tutorial>
<plan>
{self.plan}
</plan>
<pseudocode>
{self.pseudocode}
</pseudocode>
"""


class Solution(Analysis):
    source_code: str = Field(
        ..., description="Valid Python3 sourcecode to solve the problem."
    )

    @property
    def as_xml(self) -> str:
        return f"""
<root>
{super().as_xml}
<source_code>
{self.source_code}
</source_code>
</root>
"""


class Reflection(BaseModel):
    reflection: str = Field(
        ...,
        description="Reflection on the problem, your solution, and the correct answer.",
    )
    keywords: List[str] = Field(
        ...,
        description="Keywords that describe the type of your errors from most general to most specific.",
    )
    step_by_step_solution: str = Field(
        ...,
        description="Step by step solution to the problem based on your knowledge of the correct answer.",
    )
    instructions: List[str] = Field(
        ...,
        description="Detailed instructions to help you correctly solve this problem in the future.",
    )
    general_advice: List[str] = Field(
        ...,
        description="General advice to help you solve similar types of problems in the future.",
    )

    @property
    def as_xml(self) -> str:
        return f"""
<root>
<reflection>
{self.reflection}
</reflection>
<keywords>
{self.keywords}
</keywords>
<step_by_step_solution>
{self.step_by_step_solution}
</step_by_step_solution>
<instructions>
{self.instructions}
</instructions>
<general_advice>
{self.general_advice}
</general_advice>
</root>
"""


def format_example(example: dict) -> str:
    formatted_doc = f"""
<problem>
<problem_statement>
{example['description']}
</problem_statement>
</problem>
<solution>
{example['code']}
</solution>
"""
    return formatted_doc


def format_examples(examples: List[dict], analyses: List[Analysis]) -> str:
    def format_question(example: dict) -> str:
        return f"""
<problem>
<problem_statement>
{example['description']}
</problem_statement>
</problem>
"""

    def format_solution(analysis: Analysis, example: dict) -> str:
        return f"""
<root>
{analysis.as_xml}
<source_code>
{example['code']}
</source_code>
</root>
"""

    messages = ""
    for example, analysis in zip(examples, analyses):
        messages += f"\n<example>{format_question(example)}\n{format_solution(analysis, example)}</example>\n"
    return messages.strip()
