from .mmlu import MMLU, MMLUInstanceDataset
from .base import Dataset, DataLoader

from typing import Tuple, Callable
from textgrad import Variable
from textgrad.engine import EngineLM

AVAILABLE_DATASETS = [
    "BBH_object_counting",
    "BBH_tracking_shuffled_objects_five_objects",
    "temporal_sequences",
    "GSM8K_DSPy",
]

AVAILABLE_INSTANCE_DATASETS = [
    "MMLU_machine_learning",
    "MMLU_college_physics",
    "GPQA_diamond"
]

def parse_multiple_choice_answer(text: str) -> Optional[str]:
    """
    A robust helper function to extract the last multiple-choice option letter
    from a given text. It first searches for the `(X)` format, and falls back
    to finding a standalone letter if the primary format is not found.
    """
    try:
        last_line = text.strip().upper().split('\n')[-1]

        matches_with_parens = re.findall(r'\(([A-L])\)', last_line)
        if matches_with_parens:
            return matches_with_parens[-1]

        option_letters = {chr(i) for i in range(ord('A'), ord('Z') + 1)}
        for token in reversed(last_line.split()):
            purified_token = ''.join([char for char in token if char in option_letters])
            if len(purified_token) == 1:
                return purified_token

        return None

    except (ValueError, IndexError):
        return None


def multiple_choice_equality_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    """
    A specialized function to parse and evaluate multiple-choice answers,
    similar in structure to string_based_equality_fn. It extracts and
    compares only the letter of the chosen option.
    """
    predicted_answer = parse_multiple_choice_answer(str(prediction.value))
    correct_answer = parse_multiple_choice_answer(str(ground_truth_answer.value))
    return int(predicted_answer is not None and predicted_answer == correct_answer)


def load_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs) -> Tuple[Dataset, Dataset, Callable]:
    """
    Args:
        task_name: the name of the task to evaluate
        evaluation_api: the engine to use for evaluation, if needed
    """
    if "object_counting" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard, string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn

    elif ("tracking_shuffled_objects" in task_name) or ("temporal_sequences" in task_name):
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard, string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        task_name_cleaned = task_name[4:]
        train_set = BigBenchHard(task_name_cleaned, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name_cleaned, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name_cleaned, split="test", *args, **kwargs)

        eval_fn = StringBasedFunction(multiple_choice_equality_fn,
                                      function_purpose="Extracts and evaluates the final multiple-choice answer.")
        return train_set, val_set, test_set, eval_fn

    elif "BBH" in task_name:
        from textgrad.loss import MultiFieldTokenParsedEvaluation
        from .big_bench_hard import BigBenchHard
        task_name = task_name[4:]
        train_set = BigBenchHard(task_name, split="train", *args, **kwargs)
        val_set = BigBenchHard(task_name, split="val", *args, **kwargs)
        test_set = BigBenchHard(task_name, split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Reasoning and prediction from the language model"
        ]

        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and reasoning with the final prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = Variable(evaluation_instruction, requires_grad=False,
                                    role_description="evaluation instruction for the task")
        eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=evaluation_api,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"]
        )

        return train_set, val_set, test_set, eval_fn

    elif task_name == "GSM8K_DSPy":
        from textgrad.tasks.gsm8k import GSM8K_DSPy
        from .big_bench_hard import string_based_equality_fn
        from textgrad.autograd.string_based_ops import StringBasedFunction
        evaluation_instruction = "Below is a prediction we got for a question answering task, and the correct final answer. Is the final answer correct? Say only 1 (yes) or 0 (no). Return 1 if and only if the final answer is correct. Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        system_prompt = Variable(
            "You are a language model that evaluates the accuracy of a prediction for a mathematical question answering task. Only call a prediction accurate if it is the same as the ground truth answer.",
            requires_grad=False, role_description="system prompt for the evaluation")
        # Should we do train/test like this?
        train_set = GSM8K_DSPy(split="train", *args, **kwargs)
        val_set = GSM8K_DSPy(split="val", *args, **kwargs)
        test_set = GSM8K_DSPy(split="test", *args, **kwargs)
        role_descriptions = [
            "Question for the task",
            "Ground truth answer",
            "Prediction from the language model"
        ]
        fn_purpose = "The runtime of string-based function that checks if the prediction is correct."
        eval_fn = StringBasedFunction(string_based_equality_fn, function_purpose=fn_purpose)
        return train_set, val_set, test_set, eval_fn

    else:
        raise ValueError(f"Task {task_name} not found.")


def load_instance_task(task_name: str, evaluation_api: EngineLM, *args, **kwargs):
    if "MMLU_" in task_name:
        subset = task_name[5:]
        test_set = MMLUInstanceDataset(evaluation_api=evaluation_api, subset=subset, split="test", *args, **kwargs)
        return test_set
    elif "GPQA" in task_name:
        from .gpqa import GPQAInstanceDataset
        test_set = GPQAInstanceDataset(evaluation_api=evaluation_api, subset=task_name.lower(), *args, **kwargs)
        return test_set
    else:
        raise ValueError(f"Instance task {task_name} not found.")