# %% [markdown]
# <a href="https://colab.research.google.com/github/wandb/aihackercup/blob/main/one_shot_solver.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# 

# %% [markdown]
# <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
# 
# # W&B Lighting Competition - AI Hacker Cup 
# 
# </a>
# 
# [Weights & Biases](https://wandb.ai/site?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup) are running a 7-day Lightning Competition focussed on solving practice problems for the  [2024 NeurIPS AI Hacker Cup](https://hackercupai.github.io/) challenge.
# 
# #### Goal
# The goal is to try and solve all 5 of the 2023 practice questions for the AI Hacker Cup using MistralAI's models. We’re offering free MistralAI api access via the code in this colab to get people started.
# 
# #### Competition GitHub
# The competition [repo here](https://github.com/wandb/aihackercup) contains this colab, the code for the Code Generation Agent and the details on how to make a submission and the competition rules. Note that to run this notebook you'll need to be running it with a T4 GPU (15GB) or larger as the embedding model is run locally.
# 
# #### Discord
# You can join the official NeurIPS AI Hacker Cup [discord here](discord.gg/wWeN9hTH32) to share ideas and discuss winning solutions.
# 
# ## Prizes
# 
# Weights & Biases are giving away a pair of Meta Ray-Ban Smart Glasses for the first individual to submit code that solves:
# - 3 out of 5 correct solutions
# - 4 out of 5 correct solutions
# - 5 out of 5 correct solutions
# 
# (i.e. in total 3 pairs of sunglasses to give away)
# 
# ## Entry Submissions, Rules & Deadline
# 
# See the [competition README](https://github.com/wandb/aihackercup) for how to make a submissions the the competition rules.

# %% [markdown]
# ## W&B Weave
# 
# [W&B Weave](https://weave-docs.wandb.ai/tutorial-eval?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup) is used in this competition to run the evaluations. It is a lightweight toolkit for tracking and evaluating LLM applications, built by Weights & Biases. 
# 
# <img src="https://raw.githubusercontent.com/wandb/weave/master/docs/static/img/evals-hero.png" width="800" height="450">
# 
# If you want to learn more about Weave, you can [get started](https://weave-docs.wandb.ai/quickstart?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup) by decorating Python functions with `@weave.op`.

# %% [markdown]
# # A simple one-shot solver for the AI Hacker Cup 2024 Qualification Round

# %% [markdown]
# ## Setup 

# %% [markdown]
# **Note: You need to run this cell only once**
# We will clone the starter-kits repo
# Set the rag folder as our working directory
# and install the dependencies for the project.
# 
# **You can comment out the cell after you have run it once.**

# %%
# Clone the starter-kits repo
!git clone https://github.com/wandb/aihackercup
# Change directory to the rag folder. Running the next line twice in the same session will raise an error.
%cd aihackercup
# Install dependencies
!pip install -r requirements.txt -qq

# %% [markdown]
# To run this colab, create a [free Weights & Biases (W&B) account here](https://wandb.ai/site?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup) and then copy your API key from https://wandb.ai/authorize into the input box below when requested.

# %%
import os
import weave

WEAVE_PROJECT = "ai-hacker-cup"
weave_client = weave.init(WEAVE_PROJECT)

# %%
# Select MistralAI models used depending if you want a fast or strong LLM
# You can see the full range of MistralAI models here: https://docs.mistral.ai/getting-started/models/
FAST_LLM = "open-mistral-nemo-2407"
STRONG_LLM = "mistral-large-latest"

os.environ["FAST_LLM"] = STRONG_LLM  # We'll use stong model everywhere
os.environ["STRONG_LLM"] = STRONG_LLM

# URL for the MistralAI api we'll be using
os.environ["BASE_URL"] = "http://195.242.25.198:8000/v1"
os.environ["API_KEY"] = "dummy_key"

# Set the max tokens for the models and how many parallel requests to make in Weave Evaluations
os.environ["MAX_TOKENS"] = "4096"
os.environ["WEAVE_PARALLELISM"] = "2"

# %%
import asyncio
import logging

# Start of workout
from utils import Problem, async_client, format_response, check_correctness

# %% [markdown]
# ## Challenges Dataset
# We will use the **practice** dataset from the **2023** [HackerCup dataset](https://huggingface.co/datasets/hackercupai/hackercup).
# 
# We have already processed the dataset and saved it as a [`weave.Dataset`](https://weave-docs.wandb.ai/guides/core-types/datasets/?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup). You can either use the Dataset by running the next cell or download the dataset using the instructions below.
# 
# We will use this challenge dataset to load some practice problems and solutions from the HackerCup dataset and evaluate our agents on it.

# %%
# get dataset
practice_dataset_uri = "weave:///parambharat/hackercup/object/practice_dataset:R35fXf9N3FE2IOesg7bRPaPAxiE9YbpirhXO9HcHs8w"
problems_dataset = weave.ref(practice_dataset_uri).get().rows[:]
problems = list(map(lambda x: Problem(**x), problems_dataset))

# %% [markdown]
# Let's define what we expect as a solution:

# %%
from pydantic import BaseModel, Field

class Solution(BaseModel):
    core_question: str = Field(..., description="Core question of the problem")
    problem_solving_info: str = Field(..., description="Problem-solving information related to the core question")
    plan: str = Field(..., description="Step by step plan to solve the problem")
    pseudocode: str = Field(..., description="Pseudocode to solve the problem")
    source_code: str = Field(..., description="Valid Python3 sourcecode to solve the problem.")

# %% [markdown]
# ## One Shot Solver
# 
# Here we define the One Shot Solver pipeline which:
# - takes a problem as input
# - generates a solution using a large language model
# - executes the generated code
# - checks if the executed code produces the correct output
# - returns the solution and test report
# The solver uses a system prompt and template to guide the LLM in generating
# a step-by-step solution, including core question extraction, problem-solving plan,
# pseudocode, and final Python code.
# 

# %%
system_prompt = """
You are a world-class competitive programmer tasked with solving a programming problem. 
You will be provided with a problem statement, and you need to create a Python3 solution for it. 
Your task it to develop a winning solution to the problem in Python3 programming language.
You will do this in a step-by-step manner.

Step 1: Extract the core question and the problem-solving information from the problem statement.
Step 2: Generate a step by step plan to solve the problem.
Step 3: Generate the pseudocode to solve the problem.
Step 4: Write the final solution in Python3 programming language to solve the problem.

Competition Guidelines:
    a. Do not use any external libraries; stick to Python 3 standard library
    b. Handle input and output using standard input/output (stdin/stdout)
    c. Use helper functions to improve readability of the code.
    c. Use the `input()` function to take input from stdin and print the output to stdout.
    d. Do not add extra print statements otherwise it will fail the test cases.
    e. Make sure your code passes all potential test cases, including edge cases
    f. Follow the input/output format specified in the problem statement and the sample test cases."""

prompt_template = """
Let's think step by step to solve the problem:

Problem: 
{problem_description}

Input: 
{sample_input}

Output: 
{sample_output}
"""

# %%

@weave.op
async def one_shot_solver(
    problem: Problem, 
    llm_model: str,
    system_prompt: str, 
    prompt_template: str,
    temperature: float = 0.7,
    timeout: int = 10
) -> str:
    logging.info(f"Solving problem: {problem.problem_name}")

    # call model one first time to get the code
    logging.info("Calling model to solve the problem")
    model_output = await async_client.chat.completions.create(
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_template.format(
                problem_description=problem.problem_description,
                sample_input=problem.sample_input,
                sample_output=problem.sample_output)}
                ],
        temperature=temperature,
        response_model=None
    )

    out = model_output.choices[0].message.content

    # extract code from the response
    logging.info("Formatting the response")
    solution = await format_response(out, Solution)

    # check if the code is correct
    logging.info("Checking if the code is correct")
    test_report = await check_correctness(
        solution.source_code,
        problem.sample_input,
        problem.sample_output,
        timeout=timeout,
    )

    return {"solution": solution, "test_report": test_report}

# %% [markdown]
# # Evaluation

# %% [markdown]
# Now we are ready to evaluate against the expected solutions.
# 
# ### Create a Weave Model
# First we create a Weave ["Model"](https://weave-docs.wandb.ai/guides/core-types/models?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup), which has a `predict` function that Weave Evaluations will call to generate a solution. It also has various attributes that we can set to adjust the behaviour of our pipeline.

# %%
class OneShotSolver(weave.Model):
    code_execution_timeout: int = 30
    llm_model: str = STRONG_LLM
    system_prompt: str = system_prompt
    prompt_template: str = prompt_template
    temperature: float = 0.7

    @weave.op
    async def predict(self, problem: dict):
        return await one_shot_solver(
            problem=Problem(**problem), 
            llm_model=self.llm_model,
            system_prompt=self.system_prompt, 
            prompt_template=self.prompt_template, 
            timeout=self.code_execution_timeout,
            temperature=self.temperature
        )

# %% [markdown]
# ### Create the Evals Dataset and a Scorer

# %% [markdown]
# We expect the output of the "test_report" from our agent above to be `"passed"` if the solution is correct. You can think of `expected_result` in the `evals_dataset` as the label that the `test_report` from our solver needs to return in order to ensure the generated solution is correct. In this case the scoring is actually happening in our agentic pipeline as the agent needs to know the result so it can decide whether or not to retry.
# 
# Weave Evaluations expects data formatted as a list of dictionaries for the evaluation dataset. We dump `problem` as a dictionary.

# %%
evals_dataset = [{"problem": problem.model_dump(), "expected_result": "passed"} for problem in problems]

# %% [markdown]
# Weave Evaluations use a scorer function that returns a metric and its result in a dict. Here we define a metric that checks if the code generated by agent passed the test case

# %%
@weave.op
def scorer(expected_result: str, model_output: dict) -> dict:
    if model_output is None or model_output["test_report"].status is None:
        return {"solution_passed": False}
    return {"solution_passed": expected_result == model_output["test_report"].status} # check if the test_report status == passed

# %%
model = OneShotSolver()

evaluator = weave.Evaluation(dataset=evals_dataset, scorers=[scorer], trials=1)

results = await evaluator.evaluate(model)


