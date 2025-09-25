import textgrad as tg
from textgrad.engine import EngineLM
from textgrad import BlackboxLLM
import concurrent.futures
from typing import List, Any, Tuple
import json

def get_eval_output(x, y, model, eval_fn):
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
    except Exception:
        eval_output_variable = eval_fn([x, y, response])
    return eval_output_variable

def _generate_adaptive_perturbations(system_prompt_text: str, k: int, perturbation_engine: EngineLM) -> List[
    str]:

    prompt_for_analysis = f"""You are a specialized Prompt Architecture Analyst. Your task is to analyze a system prompt and decompose it into a three-tier hierarchy of components based on their sensitivity.

**Definition of Tiers:**
- **Tier 1: Constraint Layer (High Sensitivity)**: Non-negotiable rules that define success or failure. Changing these will likely break the prompt's core function or format. (e.g., format rules, core task definition, absolute prohibitions).
- **Tier 2: Method Layer (Medium Sensitivity)**: Guidelines on "how" to perform the task. Changing these affects the quality and reasoning path, but not task completion itself. (e.g., 'think step by step', process instructions).
- **Tier 3: Style Layer (Low Sensitivity)**: Persona, tone, and other stylistic elements. Changing these affects the prompt's personality, not its logic. (e.g., 'You are a helpful assistant', politeness).

**Original Prompt:**
"{system_prompt_text}"

Your output MUST be a single, valid JSON object with three keys: "constraint_layer", "method_layer", and "style_layer". Each key must have a list of strings as its value.

Example of a valid output format:
{{
  "constraint_layer": ["Provide only the final numerical answer."],
  "method_layer": ["Think step by step."],
  "style_layer": ["You are a helpful counting assistant."]
}}

Now, provide the three-tier JSON analysis for the original prompt:
"""

    analysis_response = perturbation_engine(prompt_for_analysis)
    try:
        analysis_text = analysis_response[analysis_response.find('{'):analysis_response.rfind('}') + 1]
        analysis_json = json.loads(analysis_text)
    except Exception as e:
        analysis_text = f"Could not parse analysis. Using original prompt as context: {system_prompt_text}"

    prompt_for_adaptive_perturbation = f"""You are an expert in semantics and creative writing. The goal is to explore closely related, but potentially worse-performing, variations of the text.
**1. Original Prompt:**
"{system_prompt_text}"

**2. Three-Tier Sensitivity Analysis:**
{analysis_text}

YOUR TASK & RULES: 

Your generated versions MUST adhere to these rules: 

    1. Maintain Core Intent (Global Constraint): All perturbed versions MUST maintain the core intent of the original prompt. The goal is to create semantically close variations to find weaknesses, not to write a new prompt.

    2.Targeted & Differentiated Perturbation: Your changes should be targeted, and their degree must be based on the component's sensitivity from the analysis:

        - For **Tier 1 (Constraint Layer - High Sensitivity)** components, apply only MINIMAL and SUBTLE changes (e.g., synonym swaps like 'only' -> 'just', slight rephrasing). These are fragile and require careful stress-testing.

        - For **Tier 2 (Method Layer - Medium Sensitivity)** components, you can apply MODERATE changes (e.g., rephrasing the reasoning process, altering the sequence of steps).

        - For **Tier 3 (Style Layer - Low Sensitivity)** components, you have the most freedom. Apply CREATIVE and DIVERSE changes (e.g., completely changing the persona, tone, or conversational style).

    3. No Invalid Information: Do not introduce irrelevant information, contradictions, or factual errors.

    4.Strict Output Format: The output MUST be a valid Python list of strings. Your response should contain ONLY this Python list and nothing else. Do not add any introductory text, explanations, or markdown formatting like ```python.

Example of a valid output format for k=3:
["version one", "version two", "version three"]

Now, provide the {k} targeted, perturbed versions for the original text, strictly following all the rules above:
"""

    perturbation_response = perturbation_engine(prompt_for_adaptive_perturbation)

    try:
        match = re.search(r"\[.*\]", perturbation_response, re.DOTALL)
        if not match:
            raise ValueError("No Python list found in the LLM response.")

        list_string = match.group(0)
        perturbed_texts = ast.literal_eval(list_string)

        if not isinstance(perturbed_texts, list) or not all(isinstance(s, str) for s in perturbed_texts):
            raise ValueError("LLM did not return a valid list of strings.")

        return perturbed_texts[:k]

    except Exception as e:
        variations_text = [p.strip() for p in perturbation_response.split('\n\n') if p.strip()]
        if len(variations_text) >= 1:
            return variations_text[:k]

    return []

def find_worst_neighbor_prompt_atare(
        system_prompt: tg.Variable,
        k_perturbations: int,
        batch_data: List[Tuple[str, Any]],
        model_api: EngineLM,
        eval_fn: Any,
        perturbation_engine: EngineLM,
        num_threads: int
) -> str:
    """
    Finds and returns the text of the worst-performing neighbor prompt on a given training batch.
    """
    perturbed_prompts_text = _generate_adaptive_perturbations(system_prompt.value, k_perturbations,
                                                                        perturbation_engine)

    if not perturbed_prompts_text:
        return system_prompt.value

    perturbed_prompts_vars = [tg.Variable(p_text, role_description="Perturbed prompts after update") for p_text in perturbed_prompts_text]

    perturbation_results = []
    batch_x, batch_y = zip(*batch_data)

    for i, p_var in enumerate(perturbed_prompts_vars):
        temp_model = BlackboxLLM(model_api, p_var)
        losses_for_p = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_eval_output, x, y, temp_model, eval_fn) for x, y in zip(batch_x, batch_y)]
            for future in concurrent.futures.as_completed(futures):
                losses_for_p.append(future.result())

        total_loss_for_p = tg.sum(losses_for_p)
        failure_count = total_loss_for_p.value.count('0')
        perturbation_results.append({"prompt_text": p_var.value, "failure_count": failure_count})

    if not perturbation_results:
        return system_prompt.value

    worst_result = max(perturbation_results, key=lambda r: r["failure_count"])

    return worst_result["prompt_text"]

def find_worst_perturbation_and_apply_gradient_atare(
        system_prompt: tg.Variable,
        k_perturbations: int,
        batch_data: List[Tuple[str, str]],
        model_api: EngineLM,
        eval_fn: Any,
        perturbation_engine: EngineLM,
        num_threads: int,
) -> None:

    perturbed_prompts_text = _generate_adaptive_perturbations(system_prompt.value, k_perturbations,
                                                                       perturbation_engine)
    perturbed_prompts_vars = [tg.Variable(p_text,
                                          role_description="An adaptively perturbed prompt variable (ASAM).",
                                          predecessors=[system_prompt])
                              for p_text in perturbed_prompts_text]

    perturbation_results = []
    batch_x, batch_y = zip(*batch_data)

    original_model = BlackboxLLM(model_api, system_prompt)
    original_losses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(get_eval_output, x, y, original_model, eval_fn) for x, y in zip(batch_x, batch_y)]
        for future in concurrent.futures.as_completed(futures):
            original_losses.append(future.result())
    original_variable_loss = tg.sum(original_losses)
    original_variable_loss_str = original_variable_loss.value.replace('\n', ', ')

    for i, p_var in enumerate(perturbed_prompts_vars):
        temp_model = BlackboxLLM(model_api, p_var)
        losses_for_p = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(get_eval_output, x, y, temp_model, eval_fn) for x, y in zip(batch_x, batch_y)]
            for future in concurrent.futures.as_completed(futures):
                losses_for_p.append(future.result())

        total_loss_for_p = tg.sum(losses_for_p)
        perturbation_results.append({
            "prompt_var": p_var,
            "total_loss": total_loss_for_p
        })
        loss_str_for_print = total_loss_for_p.value.replace('\n', ', ')
    worst_result = max(perturbation_results, key=lambda r: r["total_loss"].value.count('0'))
    worst_loss_variable = worst_result["total_loss"]
    worst_prompt_var = worst_result["prompt_var"]

    worst_score = worst_result["total_loss"].value.count('0')
    worst_loss_variable.backward()
    captured_gradients = worst_prompt_var.gradients
    captured_gradients_context = worst_prompt_var.gradients_context

    if not captured_gradients:
        return

    for grad in captured_gradients:
        system_prompt.gradients.add(grad)
        original_context_dict = captured_gradients_context.get(grad)
        if original_context_dict:
            new_context_dict = original_context_dict.copy()
            conversation_context = new_context_dict.get("context", "")
            performance_explanation = (
                "Here is the performance analysis of the original prompt and a perturbed version on the current training batch.\n"
                "The loss string shows the outcome for each sample in the batch, where '1' indicates a correct answer and '0' indicates an incorrect answer."
            )
            perturbed_variable_loss_str = worst_result["total_loss"].value.replace('\n', ', ')
            enhanced_context_string = (
                f"{performance_explanation}\n\n"
                f"<ORIGINAL_VARIABLE> {system_prompt.value} </ORIGINAL_VARIABLE>\n"
                f"<ORIGINAL_VARIABLE_LOSS> {original_variable_loss_str} </ORIGINAL_VARIABLE_LOSS>\n\n"
                f"<PERTURBED_VARIABLE> {worst_prompt_var.value} </PERTURBED_VARIABLE>\n"
                f"<PERTURBED_VARIABLE_LOSS> {perturbed_variable_loss_str} </PERTURBED_VARIABLE_LOSS>\n\n"
                f"{conversation_context}"
            )
            new_context_dict["context"] = enhanced_context_string
            new_context_dict["original_variable_value"] = system_prompt.value
            new_context_dict["perturbed_variable_value"] = worst_prompt_var.value
            system_prompt.gradients_context[grad] = new_context_dict
        else:
            system_prompt.gradients_context[grad] = None



