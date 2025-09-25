import textgrad as tg
from textgrad.engine import EngineLM
from textgrad import BlackboxLLM
import concurrent.futures
from typing import List, Any, Tuple

def get_eval_output(x, y, model, eval_fn):
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
    except:
        eval_output_variable = eval_fn([x, y, response])
    return eval_output_variable

def _generate_perturbations(system_prompt_text: str, k: int, perturbation_engine: EngineLM) -> List[str]:
    prompt_for_perturbation = f"""You are an expert in semantics and creative writing. Your task is to generate {k} slightly different versions of the following text.

These perturbed versions must adhere to these rules:
1.  Maintain the core intent and theme of the original text.
2.  The degree of perturbation should be small. For example, you can replace a few non-essential words, make minor adjustments to sentence structure, or add/remove a few descriptive words.
3.  Do not introduce irrelevant information or factual errors.
4.  The goal is to explore closely related, but potentially worse-performing, variations of the text.

The output should be a Python-style list of strings, with each string being one perturbed version.

Original Text:
"{system_prompt_text}"

The output MUST be a valid Python list of strings. Your response should contain ONLY this Python list and nothing else. Do not add any introductory text, explanations, or markdown formatting like ```python.

Example of a valid output format for k=3:
["version one", "version two", "version three"]

Now, provide the {k} perturbed versions for the original text:
"""

    response = perturbation_engine(prompt_for_perturbation)
    try:
        perturbed_texts = eval(response)
        if not isinstance(perturbed_texts, list) or not all(isinstance(s, str) for s in perturbed_texts):
            raise ValueError("LLM did not return a valid list of strings.")
        return perturbed_texts[:k]
    except Exception as e:
        return [line.strip() for line in response.split('\n') if line.strip()][:k]

def find_worst_neighbor_prompt_tare(
        system_prompt: tg.Variable,
        k_perturbations: int,
        batch_data: List[Tuple[str, Any]],
        model_api: EngineLM,
        eval_fn: Any,
        perturbation_engine: EngineLM,
        num_threads: int
) -> str:
    perturbed_prompts_text = _generate_perturbations(system_prompt.value, k_perturbations,
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

def find_worst_perturbation_and_apply_gradient(
        system_prompt: tg.Variable,
        k_perturbations: int,
        batch_data: List[Tuple[str, str]],
        model_api: EngineLM,
        eval_fn: Any,
        perturbation_engine: EngineLM,
        num_threads: int,
) -> None:

    perturbed_prompts_text = _generate_perturbations(system_prompt.value, k_perturbations, perturbation_engine)
    perturbed_prompts_vars = [tg.Variable(p_text,
                                          role_description="A perturbed prompt variable.",
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



