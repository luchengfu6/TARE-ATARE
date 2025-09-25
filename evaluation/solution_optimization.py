import re
import json
import argparse
import concurrent
from tqdm import tqdm
import numpy as np
import TARE as tr
from TARE.autograd import solution_atare_perturbations

from statistics import multimode
import textgrad as tg
from textgrad.tasks import load_instance_task


def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="GPQA_diamond", help="The task to evaluate the model on.")
    parser.add_argument("--engine", type=str, default="gpt-4o", help="The API to use for evaluation.")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="The maximum number of iterations of test-time updates.")
    parser.add_argument("--num_threads", type=int, default=10, help="The number of threads to use for evaluation.")
    parser.add_argument("--k_perturbations", type=int, default=3,
                        help="The number of perturbations to generate for TARE/ATARE.")
    return parser.parse_args()


class MajorityVoting:
    def __init__(self):
        pass

    def __call__(self, predictions):
        ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer\s*:\s*([A-D])"
        pred_labels = []
        for pred in predictions:
            match = re.search(ANSWER_PATTERN_MULTICHOICE, pred.value)
            extracted_answer = match.group(1) if match else None
            pred_labels.append(extracted_answer)

        if not any(pred_labels):
            return tg.Variable("Answer: N/A", role_description="Majority ensemble")

        modes = multimode(filter(None, pred_labels))
        return tg.Variable(f"Answer: {modes[0]}", role_description="Majority ensemble")


def get_zeroshot_answer(question):
    """Getting the zero-shot answer from an LLM without optimizing the response at test time."""
    STARTING_SYSTEM_PROMPT = (
            "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
            + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
    )
    system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, requires_grad=False,
                                role_description="system prompt to the language model")
    model = tg.BlackboxLLM(llm_engine, system_prompt)
    response = model(tg.Variable(question, requires_grad=False, role_description="question to the language model"))
    return response


def run_test_time_training(sample):
    performance_history = []
    question, answer, test_time_objective, instance_eval_fn = sample
    zero_shot_response = get_zeroshot_answer(question)

    instance_var = tg.Variable(zero_shot_response.value,
                               requires_grad=True,
                               role_description="creative and precise solution and the prediction for the multiple choice question")

    performance_history.append(int(instance_eval_fn(instance_var)))

    optimizer = tr.SAMTextualGradientDescent(engine=llm_engine, parameters=[instance_var],
                                                 constraints=[
                                                     "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD."])

    predictions = [tg.Variable(instance_var.value, role_description=instance_var.role_description)]
    perturbation_engine = llm_engine
    for i in range(args.max_iterations):
        optimizer.zero_grad()
        solution_atare_perturbations.solution_atare_backward_pass(question, instance_var, test_time_objective, instance_eval_fn, perturbation_engine, args.k_perturbations)
        optimizer.step()
        performance_history.append(instance_eval_fn(instance_var))
        predictions.append(tg.Variable(
            instance_var.value,
            role_description=instance_var.role_description
        ))
    ensembled_prediction = ensembler(predictions)
    performance_history.append(instance_eval_fn(ensembled_prediction))
    predictions.append(ensembled_prediction)
    return performance_history, predictions, question, answer


args = config()
llm_engine = tg.get_engine(engine_name=args.engine)
tg.set_backward_engine(llm_engine, override=True)
test_set = load_instance_task(args.task, evaluation_api=llm_engine)
ensembler = MajorityVoting()

all_solutions = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    futures = []
    for _, sample in enumerate(test_set):
        future = executor.submit(run_test_time_training, sample)
        futures.append(future)

    all_history = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0):
        performance_history, predictions, question, answer = future.result()
        all_solutions[question] = {
            "predictions": [p.value for p in predictions],
            "answer": answer,
            "performance_history": performance_history
        }
        all_history.append(performance_history)

if all_history:
    summary_accuracy = np.array(all_history).mean(axis=0).tolist()
    print("Accuracy Trajectory (Initial, Iter 1, Iter 2, ..., Final Ensembled):")
    print(summary_accuracy)
    all_solutions["summary_accuracy_trajectory"] = summary_accuracy
else:
    print("No results to analyze.")

engine_name_sanitized = args.engine.replace('/', '_')
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"./results/solution/{args.task}_{engine_name_sanitized}_results_{current_time_str}.json"

print(f"\nSaving results to: {output_filename}")
with open(output_filename, "w") as f:
    json.dump(all_solutions, f, indent=4)