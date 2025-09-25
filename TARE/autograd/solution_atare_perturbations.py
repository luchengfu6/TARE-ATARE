import json
import random
import re
import textgrad as tg
from textgrad.variable import Variable
from textgrad.loss import MultiChoiceTestTime
import ast
import os, re

def _diagnose_flaw(solution_var: Variable, perturbation_engine) -> str:
    prompt = f"""You are a world-class expert in logic and science. The following solution is INCORRECT. Your task is to deeply analyze its reasoning and clearly describe the core 'cognitive trap' or 'flawed reasoning path' that led to the wrong conclusion.
Incorrect Solution to Diagnose (S_t):
"{solution_var.value}"
Please provide your detailed analysis of its flawed reasoning path:"""
    flaw_analysis = perturbation_engine(prompt)
    return flaw_analysis

def _generate_flaw_variations(question: str, solution_var: Variable, flaw_analysis: str, k: int,
                              perturbation_engine) -> list:
    prompt = f"""You are a creative AI that can mimic different thinking styles. You will receive an incorrect solution and an analysis of its core flaw.
Your task is to generate {k} new, distinct, but equally flawed solutions for the given "Problem Context".
**Guiding Principle:** Treat the solution as a reasoning chain. The correct reasoning *before* the identified flaw is the high-sensitivity backbone that MUST be preserved. Your task is to vary the expression of the flaw itself (the low-sensitivity target).
All new solutions MUST:
1. Commit the same type of core error as described in the 'Flaw Analysis'.
2. But, use different phrasing, examples, or intermediate steps to express this error. The goal is to explore different deceptive manifestations of this single cognitive trap.
3. Ensure that the final conclusion and the chosen answer letter LOGICALLY FOLLOW from your flawed reasoning.
4. You MUST choose your final answer from the available options in the "Problem Context". Do not invent new options.

--- Problem Context ---
{question}

--- Original Incorrect Solution (S_t) ---
{solution_var.value}

--- Flaw Analysis Report ---
{flaw_analysis}

The output MUST be a valid Python list of strings. Your response should contain ONLY this Python list and nothing else. Do not add any introductory text, explanations, or markdown formatting like ```python.

Example of a valid output format for k=3:
["Flawed solution text 1", "Flawed solution text 2", "Flawed solution text 3"]
Now, provide the {k} flawed solutions based on the information above:
"""
    response = perturbation_engine(prompt)
    try:
        match = re.search(r"\[.*\]", response, re.DOTALL)
        if not match:
            raise ValueError("No Python list found in the LLM response.")
        list_string = match.group(0)
        sanitized_list_string = list_string.replace('\\', '\\\\')
        variations_text = ast.literal_eval(sanitized_list_string)

        if not isinstance(variations_text, list) or not all(isinstance(s, str) for s in variations_text):
            raise ValueError("LLM did not return a valid list of strings.")

        if len(variations_text) > 0:
            return [tg.Variable(text,
                                requires_grad=True,
                                role_description=f"Flaw Variation {i + 1}",
                                predecessors=[solution_var])
                    for i, text in enumerate(variations_text[:k])]
        return []

    except Exception as e:
        variations_text = [p.strip() for p in response.split('\n\n') if p.strip()]
        if len(variations_text) > 1:
            return [tg.Variable(text,
                                requires_grad=True,
                                role_description=f"Flaw Variation {i + 1}",
                                predecessors=[solution_var])
                    for i, text in enumerate(variations_text[:k])]

    return []

def _find_worst_via_tournament(flaw_variations: list, question: str, perturbation_engine) -> Variable:
    formatted_solutions = []
    for i, p_var in enumerate(flaw_variations):
        formatted_solutions.append(f"--- Solution to Evaluate {i + 1} ---\n{p_var.value}")

    solutions_block = "\n\n".join(formatted_solutions)
    tournament_prompt = f"""You are a top-tier academic review expert. Below are several INCORRECT solutions to a single problem.

Your task is to act as a judge. You must carefully evaluate each solution according to the provided SCORING RUBRIC and then identify which solution deserves the **lowest score**.

--- Problem Context ---
{question}

--- SCORING RUBRIC ---
* **1.0 (Perfect):** Flawless reasoning and a correct conclusion for the question.
* **0.7-0.9 (Minor Flaws):** The conclusion is correct, but the reasoning has minor errors.
* **0.4-0.6 (Major Flaws):** The approach is partially correct, but a major flaw leads to an incorrect conclusion.
* **0.1-0.3 (Mostly Incorrect):** The reasoning is fundamentally flawed, though related to the question's topic.
* **0.0 (Completely Wrong):** The solution is nonsensical, factually incorrect, or off-topic.

--- Candidate Incorrect Solutions ---
{solutions_block}

--- Your Verdict ---
After evaluating all candidates based on the rubric, please respond with a single JSON object, indicating the index number (starting from 1) of the solution with the **lowest score**. For example: {{"lowest_score_index": 2}}
Your response MUST contain only the JSON object.
"""
    response = perturbation_engine(tournament_prompt)

    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            verdict_json = json.loads(match.group(0))
            index = int(verdict_json["lowest_score_index"])

            if 1 <= index <= len(flaw_variations):
                return flaw_variations[index - 1]
    except (json.JSONDecodeError, KeyError, ValueError, TypeError, IndexError) as e:
        return flaw_variations[0]

    return flaw_variations[0]

def solution_atare_backward_pass(question: str,
                               instance_var: Variable,
                               test_time_objective: MultiChoiceTestTime,
                               instance_eval_fn,
                               perturbation_engine,
                               k: int):
    flaw_analysis = _diagnose_flaw(instance_var, perturbation_engine)
    if not flaw_analysis:
        loss = test_time_objective(instance_var)
        loss.backward()
        return

    flaw_variations = _generate_flaw_variations(question, instance_var, flaw_analysis, k, perturbation_engine)
    if not flaw_variations:
        loss = test_time_objective(instance_var)
        loss.backward()
        return

    worst_solution_var = _find_worst_via_tournament(flaw_variations, question, perturbation_engine)

    if worst_solution_var:
        final_worst_loss = test_time_objective(worst_solution_var)

        final_worst_loss.backward()
        captured_gradients = worst_solution_var.gradients
        captured_gradients_context = worst_solution_var.gradients_context
        for grad in captured_gradients:
            instance_var.gradients.add(grad)
            original_context_dict = captured_gradients_context.get(grad, {})
            new_context_dict = original_context_dict.copy()
            new_context_dict["original_variable_value"] = instance_var.value
            new_context_dict["perturbed_variable_value"] = worst_solution_var.value
            conversation_context = new_context_dict.get("context", "")
            enhanced_context_string = (
                f"## The Original Solution being optimized:\n<ORIGINAL_VARIABLE>\n{instance_var.value}\n</ORIGINAL_VARIABLE>\n\n"
                f"## The Adversarial Variation that prompted the feedback:\n<PERTURBED_VARIABLE>\n{worst_solution_var.value}\n</PERTURBED_VARIABLE>\n\n"
                f"## Feedback Context:\n{conversation_context}"
            )
            new_context_dict["context"] = enhanced_context_string
            instance_var.gradients_context[grad] = new_context_dict




