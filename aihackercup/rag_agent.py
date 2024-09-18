# <a href="https://colab.research.google.com/github/wandb/aihackercup/blob/main/rag_code_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# <!--- @wandbcode{rag-hackercup} -->

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

# ## W&B Weave
# 
# [W&B Weave](https://weave-docs.wandb.ai/tutorial-eval?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup) is used in this competition to run the evaluations. It is a lightweight toolkit for tracking and evaluating LLM applications, built by Weights & Biases. 
# 
# <img src="https://raw.githubusercontent.com/wandb/weave/master/docs/static/img/evals-hero.png" width="800" height="450">
# 
# If you want to learn more about Weave, you can [get started](https://weave-docs.wandb.ai/quickstart?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup) by decorating Python functions with `@weave.op`.

# # Using RAG for a Code Generation Agent
# 
# This colab demonstrates how to retrieve over a dataset of coding question-answer pairs (the [CodeContests](https://huggingface.co/datasets/deepmind/code_contests) dataset from DeepMind) in order to find simlar questions that might help our Agent generate the correct solution.
# 
# A more detailed walkthough of the approach we will use in this notebook can be found in the following **[Youtube video](https://www.youtube.com/watch?v=cObBj2UpWK8)**:
# 
# <a target="_blank" href="https://www.youtube.com/watch?v=cObBj2UpWK8">
# <img src="https://img.youtube.com/vi/cObBj2UpWK8/0.jpg" width="400" height="300">
# </a>

# ## Setup 

# **Note: You need to run this cell only once**
# We will clone the starter-kits repo
# Set the rag folder as our working directory
# and install the dependencies for the project.
# 
# **You can comment out the cell after you have run it once.**

# Clone the starter-kits repo
# git clone https://github.com/wandb/aihackercup
# Change directory to the rag folder. Running the next line twice in the same session will raise an error.
# cd aihackercup
# Install dependencies
# pip install -r requirements.txt -qq

# To run this colab, create a [free Weights & Biases (W&B) account here](https://wandb.ai/site?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup) and then copy your API key from https://wandb.ai/authorize into the input box below when requested.

import os
import weave

WEAVE_PROJECT = "ai-hacker-cup"
weave_client = weave.init(WEAVE_PROJECT)

# Select MistralAI models used depending if you want a fast or strong LLM
# You can see the full range of MistralAI models here: https://docs.mistral.ai/getting-started/models/
FAST_LLM = "open-mistral-nemo-2407"

# STRONG_LLM = "mistral-large-latest"
STRONG_LLM = "deepseek-ai/deepseek-coder-6.7b-instruct"

os.environ["FAST_LLM"] = STRONG_LLM  # We'll use stong model everywhere
os.environ["STRONG_LLM"] = STRONG_LLM

# URL for the MistralAI api we'll be using
# os.environ["BASE_URL"] = "http://195.242.25.198:8000/v1"

# Set the max tokens for the models and how many parallel requests to make in Weave Evaluations
# os.environ["MAX_TOKENS"] = "4096"
# os.environ["MAX_TOKENS"] = "8192"
os.environ["MAX_TOKENS"] = str(14480 - 2000)

os.environ["WEAVE_PARALLELISM"] = "2"



# ## Challenges Dataset
# We will use the **practice** dataset from the **2023** [HackerCup dataset](https://huggingface.co/datasets/hackercupai/hackercup).
# 
# We have already processed the dataset and saved it as a [`weave.Dataset`](https://weave-docs.wandb.ai/guides/core-types/datasets/?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup). You can either use the Dataset by running the next cell or download the dataset using the instructions below.
# 
# We will use this challenge dataset to load some practice problems and solutions from the HackerCup dataset and evaluate our agents on it.

from agent import rag_solver, rework_solution
from utils import Problem

practice_dataset_uri = "weave:///parambharat/hackercup/object/practice_dataset:R35fXf9N3FE2IOesg7bRPaPAxiE9YbpirhXO9HcHs8w"
problems_dataset = weave.ref(practice_dataset_uri).get().rows[:]
problems = list(map(lambda x: Problem(**x), problems_dataset))
problem = problems[2]  # Select the first problem

print("Sample Problem:\n\n", problem.model_dump_json(indent=2))

# #### [Alternative] Download the raw challenges dataset
# 
# You can alternatively download the full raw challenges dataset, see the README to see how.

# #### Turn on logging and asyncio for notebooks

import asyncio
import logging
from nest_asyncio import apply

apply()
logging.basicConfig(
  format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# ## Running a RAG + Reflection Agent

# ### RAG Agent with Reflection
# 
# We will combine a RAG Agent with Reflection in order to:
# 
# - Retrieve similar types of questions from the CodeContests dataset, generate a solution, reflect on the solution and test results and improve it.
# - We then use this improved solution to generate new few-shot examples and repeat the process in a loop until we converge to a solution or the iteration limit is reached.
# 
# `agent.py` contains the prompts used for analysis (`ANALYSIS_INSTRUCTIONS`), reflection (`REFLECTION_INSTRUCTIONS`) and problem solving (`SOLVER_INSTRUCTIONS`) feel free to edit them to improve the system.

from agent import REFLECTION_INSTRUCTIONS

print(REFLECTION_INSTRUCTIONS)

# ### Retriever
# 
# The code used the retrieval over the CodeContests dataset can be found in `retriever.py`. You'll see we're using the `jinaai/jina-embeddings-v2-base-code` embedding model locally as it has been trained on code. 
# 
# Here we'll initialise our retriever.

from retriever import Retriever

retriever = Retriever()

# ### RAG Solver Pipeline
# 
# Here we run the code generation pipeline which:
# - given a problem, retrieves similar problems from the CodeCompletions dataset
# - generates candidate code for problem
# - executes the code
# - checks if the executed code generates the correct solution
# - if the solution is correct, it terminates otherwise it retries for `max_iterations`
# 
# Note `code_execution_timeout`is used to limit the time available for the generated python code to execute as sometimes the code generated be recursive code that never terminates.

@weave.op
async def rag_solver_with_reflection(
    retriever: Retriever,
    problem: Problem,
    model: str = FAST_LLM,
    temperature: float = 0.7,
    max_iterations: int = 2,
    code_execution_timeout: int = 10,
):
  num_iterations = 0
  while num_iterations < max_iterations:
    rag_result = await rag_solver(
      retriever=retriever,
      problem=problem,
      timeout=code_execution_timeout,
      model=model,
      temperature=temperature,
    )
    solution, test_report = rag_result["solution"], rag_result["test_report"]
    if test_report.status == "passed":
      logger.info(f"Passing solution generated successfully for problem: {problem.problem_name}")
      return rag_result
    
    logger.info(f"Solution failed, reworking solution. Problem: {problem.problem_name}")
    rework_result = await rework_solution(
      problem=problem,
      incorrect_solution=solution,
      test_report=test_report,
      model=model,
      temperature=temperature,
      timeout=code_execution_timeout,
    )
    solution, test_report = rework_result["solution"], rework_result["test_report"]
    if test_report.status == "passed":
      logger.info(f"Re-worked solution passed for problem: {problem.problem_name}")
      return {
        "solution": solution,
        "stage": "reflection",
        "test_report": test_report,
      }
    num_iterations += 1
    logger.info(f"Re-worked solution failed, trying iteration {num_iterations}. Problem: {problem.problem_name}")
  logger.info("Failed to generate a solution after {num_iterations} iterations. Problem: {problem.problem_name}")
  return {"solution": solution, "stage": "failed", "test_report": test_report}

# Lets run the pipeline on 1 problem, **this will take about 7 minutes to complete** as it makes a lot of LLM calls and runs multiple iterations.

async def get_reflection_result(problem):
  return await rag_solver_with_reflection(
              retriever, problem, STRONG_LLM, max_iterations=2, 
              code_execution_timeout=30
  )

# reflection_result = await rag_solver_with_reflection(
# # reflection_result = rag_solver_with_reflection(
#   retriever, problem, STRONG_LLM, max_iterations=2, code_execution_timeout=30
# )

reflection_result = asyncio.run(get_reflection_result(problem=problem))

print("*" * 40 + " SOLUTION: " + "*" * 40)
print(reflection_result["solution"].source_code)
print("*" * 40 + " TEST REPORT " + "*" * 40)
print(reflection_result["test_report"])

# # Evaluation

# Now we are ready to evaluate against the expected solutions.
# 
# ### Create a Weave Model
# First we create a Weave ["Model"](https://weave-docs.wandb.ai/guides/core-types/models?utm_source=colab&utm_medium=code&utm_campaign=lightning-ai-hacker-cup), which has a `predict` function that Weave Evaluations will call to generate a solution. It also has various attributes that we can set to adjust the behaviour of our pipeline.

class RAGReflectionAgent(weave.Model):
  retriever: Retriever
  max_iterations: int = 2
  code_execution_timeout: int = 30
  model: str = STRONG_LLM
  temperature: float = 0.7

  @weave.op
  async def predict(self, problem: dict):
    return await rag_solver_with_reflection(
      self.retriever,
      Problem(**problem),
      model=self.model,
      temperature=self.temperature,
      max_iterations=self.max_iterations,
      code_execution_timeout=self.code_execution_timeout,
    )

# ### Create the Evals Dataset and a Scorer

# We expect the output of the "test_report" from our agent above to be `"passed"` if the solution is correct. You can think of `expected_result` in the `evals_dataset` as the label that the `test_report` from our solver needs to return in order to ensure the generated solution is correct. In this case the scoring is actually happening in our agentic pipeline as the agent needs to know the result so it can decide whether or not to retry.
# 
# Weave Evaluations expects data formatted as a list of dictionaries for the evaluation dataset. We dump `problem` as a dictionary.

evals_dataset = [{"problem": problem.model_dump(), "expected_result": "passed"} for problem in problems]

# Weave Evaluations use a scorer function that returns a metric and its result in a dict. Here we define a metric that checks if the code generated by agent passed the test case

@weave.op
def scorer(expected_result: str, model_output: dict) -> dict:
  if model_output is None or model_output["test_report"].status is None:
    return {"solution_passed": False}
  return {"solution_passed": expected_result == model_output["test_report"].status} # check if the test_report status == passed

# ### Run the Evaluation
# Now we instantiate the Agent and run the evaluation. Results from the evaluation will be printed in the W&B Weave UI. The WEAVE_PARALLELISM env var determines how many evaluations are run in parallel and is set at 2 by default, each can take 7 to 9 minutes.

# Evaluate the RAG reflection agent
tasks = []

LLM = STRONG_LLM
eval_temperature = 0.7

# Instantiate the agent, which is a subclass of `weave.Model`
rag_reflection_agent = RAGReflectionAgent(
  retriever=retriever, model=LLM, temperature=eval_temperature, code_execution_timeout=30
)

# Weave Evaluations take a dataset and scoring functions.
# This evaluation checks if the code generated by the agent passes
# trials can be set to run the full evaluation multiple times
evaluator = weave.Evaluation(dataset=evals_dataset, scorers=[scorer], trials=1)

# Evaluate the agent by passing it to the evaluator
# Weave Evaluations are async, so we use `asyncio.gather` to run them in parallel
# The WEAVE_PARALLELISM environment variable sets the number of evaluations to run in parallel
rag_reflection_results = evaluator.evaluate(rag_reflection_agent)
tasks.append(rag_reflection_results)

async def get_final_result(tasks):
  return await asyncio.gather(*tasks)

# rag_reflection_results = await asyncio.gather(*tasks)
# rag_reflection_results = asyncio.gather(*tasks)
rag_reflection_results = asyncio.run(get_final_result(tasks))

logger.info(rag_reflection_results)

# You will now be able to find your evaluation results in the Weights & Biases UI in the Evaluations tab. You can find a link to your Weave project under the cell above that calls `weave.init`


