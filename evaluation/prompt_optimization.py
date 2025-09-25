import os
import argparse
import concurrent
from tqdm import tqdm
import TARE as tr
import textgrad as tg
from textgrad.tasks import load_task
from TARE.autograd import tare_perturbations, atare_perturbations
import numpy as np
import random
import json

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="BBH_object_counting", help="The task to evaluate the model on.")
    parser.add_argument("--backbone_engine", type=str, default="azure-gpt4o", help="The backbone engine for textgrad.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="The model on which the prompt is optimized.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=1, help="The maximum number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_robust_validation", action="store_true", help="Whether to run robust validation or not.")
    parser.add_argument("--num_threads", type=int, default=3, help="The number of threads to use for evaluation.")
    parser.add_argument("--k_perturbations", type=int, default=3, help="The number of perturbations.")
    parser.add_argument("--k_update_perturbations", type=int, default=3, help="The number of update prompt perturbations.")
    parser.add_argument("--method", type=str, default="tare", help="Different methods of perturbations.")
    return parser.parse_args()


def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)
        return int(eval_output_parsed)


def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for _, sample in enumerate(test_set):

            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list


def validate_on_worst_neighbor_on_val_set(
        system_prompt: tg.Variable,
        prompt_before_update: str,
        batch_data: list,
        val_set: list,
        results: dict,
        args: argparse.Namespace,
        model_api,
        eval_fn
):

    if args.method == "tare":
        find_worst_prompt_fn = tare_perturbations.find_worst_neighbor_prompt_tare
    else:
        find_worst_prompt_fn = atare_perturbations.find_worst_neighbor_prompt_atare

    worst_neighbor_prompt_text = find_worst_prompt_fn(
        system_prompt=system_prompt,
        k_perturbations=args.k_update_perturbations,
        batch_data=batch_data,
        model_api=model_api,
        eval_fn=eval_fn,
        perturbation_engine=tg.get_engine(args.backbone_engine),
        num_threads=args.num_threads
    )

    worst_neighbor_prompt_var = tg.Variable(worst_neighbor_prompt_text, role_description="worst neighbor of update prompt", requires_grad=False)
    temp_model = tg.BlackboxLLM(model_api, worst_neighbor_prompt_var)
    current_robust_val_acc = np.mean(eval_dataset(val_set, eval_fn, temp_model))

    previous_robust_val_acc = results["robust_validation_acc"][-1] if results["robust_validation_acc"] else 0

    if current_robust_val_acc >= previous_robust_val_acc:
        results["robust_validation_acc"].append(current_robust_val_acc)
    else:
        system_prompt.set_value(prompt_before_update)
        results["robust_validation_acc"].append(previous_robust_val_acc)

def get_eval_output(x, y, model, eval_fn):
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(str(y), requires_grad=False, role_description="correct answer for the query")
    response = model(x)
    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
    except:
        eval_output_variable = eval_fn([x, y, response])
    return eval_output_variable


args = config()
print(vars(args))
set_seed(args.seed)
if 'llama' in args.backbone_engine:
    llm_api = tg.get_engine(engine_name=args.backbone_engine, batch_size=args.num_threads)
else:
    llm_api = tg.get_engine(engine_name=args.backbone_engine)
tg.set_backward_engine(llm_api, override=True)

train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
BBH_OTHER_PROMPT = "You will answer a reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: ($VALUE)' where VALUE is the letter of the correct option."
if ("object_counting" in args.task) or ("GSM8K" in args.task):
    STARTING_SYSTEM_PROMPT = train_set.get_task_description()
else:
    STARTING_SYSTEM_PROMPT = BBH_OTHER_PROMPT

train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
print(f'Starting system prompt: {STARTING_SYSTEM_PROMPT}')

system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT,
                            requires_grad=True,
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")

if 'llama' in args.model:
    model_api = tg.get_engine(engine_name=args.model, batch_size=args.num_threads)
else:
    model_api = tg.get_engine(engine_name=args.model)
model = tg.BlackboxLLM(model_api, system_prompt)


optimizer = tr.LATO(engine=llm_api, parameters=[system_prompt])

results = {
    "test_acc_mean": [],
    "robust_validation_acc": [],
    "test_acc": [],
    "prompt": [],
    "validation_acc": []
}
results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())
results["robust_validation_acc"].append(np.mean(results["validation_acc"][-1]))
for epoch in range(args.max_epochs):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
        pbar.set_description(f"Training step {steps + 1}, Epoch {epoch + 1}")
        prompt_before_update = system_prompt.get_value()
        optimizer.zero_grad()

        if args.method == "tare":
            tare_perturbations.find_worst_perturbation_and_apply_gradient(
                system_prompt=system_prompt,
                k_perturbations=args.k_perturbations,
                batch_data=list(zip(batch_x, batch_y)),
                model_api=model_api,
                eval_fn=eval_fn,
                perturbation_engine=tg.get_engine(args.backbone_engine),
                num_threads=args.num_threads,
            )
        elif args.method == "atare":
            atare_perturbations.find_worst_perturbation_and_apply_gradient_atare(
                system_prompt=system_prompt,
                k_perturbations=args.k_perturbations,
                batch_data=list(zip(batch_x, batch_y)),
                model_api=model_api,
                eval_fn=eval_fn,
                perturbation_engine=tg.get_engine(args.backbone_engine),
                num_threads=args.num_threads,
            )

        optimizer.step()
        if args.run_robust_validation:
            validate_on_worst_neighbor_on_val_set(
                system_prompt=system_prompt,
                prompt_before_update=prompt_before_update,
                batch_data=list(zip(batch_x, batch_y)),
                val_set=val_set,
                results=results,
                args=args,
                model_api=model_api,
                eval_fn=eval_fn
            )

        test_acc = eval_dataset(test_set, eval_fn, model)
        results["test_acc"].append([float(acc) for acc in test_acc])
        test_acc_mean = np.mean(test_acc)
        results["test_acc_mean"].append(float(test_acc_mean))
        results["prompt"].append(system_prompt.get_value())
        if steps == 11:
            break

os.makedirs("./results", exist_ok=True)
model_name = args.backbone_engine.split("/")[-1]
model_name2 = args.model.split("/")[-1]
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
filepath = f"./results/results_{args.task}_{model_name}_{model_name2}_{current_time_str}.json"

if filepath:
    directory = os.path.dirname(filepath)
    os.makedirs(directory, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)
