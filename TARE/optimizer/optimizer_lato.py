from typing import List, Union
from collections import defaultdict
from textgrad.variable import Variable
from textgrad import logger
from textgrad.engine import EngineLM
from textgrad.config import validate_engine_or_get_default
from .optimizer_prompts_lato import (
    construct_tgd_prompt as construct_sam_tgd_prompt,
    OPTIMIZER_SYSTEM_PROMPT as SAM_OPTIMIZER_SYSTEM_PROMPT,
    GRADIENT_TEMPLATE,
    GRADIENT_MULTIPART_TEMPLATE
)

def get_gradient_and_context_text(variable) -> Union[str, List[Union[str, bytes]]]:
    """For the variable, aggregates and returns
    i. the gradients
    ii. the context for which the gradients are computed.

    This is used by the optimizer.
    :return: A string containing the aggregated gradients and their corresponding context.
    :rtype: str
    """

    gradient_content = []
    for g in variable.gradients:
        if variable.gradients_context[g] is None:
            gradient_content.append(g.value)
        else:
            context = variable.gradients_context[g]
            if isinstance(context["context"], str):
                criticism_and_context = GRADIENT_TEMPLATE.format(
                    feedback=g.value, **context)
                gradient_content.append(criticism_and_context)
            elif isinstance(context["context"], list):
                context_prompt = GRADIENT_MULTIPART_TEMPLATE.format(**context, feedback=g.value)
                criticism_and_context = context["context"] + [context_prompt]
                gradient_content.extend(criticism_and_context)
            else:
                raise ValueError("Context must be either a string or a list.")
    if all(isinstance(i, str) for i in gradient_content):
        return "\n".join(gradient_content)
    else:
        return gradient_content

class Optimizer(ABC):
    """
    Base class for all optimizers.

    :param parameters: The list of parameters to optimize.
    :type parameters: List[Variable]

    :Methods:
        - zero_grad(): Clears the gradients of all parameters.
        - step(): Performs a single optimization step.
    """

    def __init__(self, parameters: List[Variable]):
        for parameter in parameters:
            if type(parameter.value) != str:
                raise NotImplementedError(
                    f"We cannot yet update multimodal content and this data type: {type(parameter.value)}. We can only evaluate gradients using multimodal models. This may change soon (looking at you, GPT-5).")
        self.parameters = parameters

    def zero_grad(self):
        """
        Clears the gradients of all parameters.
        """
        for p in self.parameters:
            p.gradients = set()

    @abstractmethod
    def step(self):
        """
        Performs a single optimization step.
        """
        pass

class LATO(Optimizer):
    """
    一个完全独立的、为类SAM优化流程设计的优化器。
    它包含了处理约束、梯度记忆等所有功能，不依赖于继承其他优化器。
    """

    def __init__(self,
                 parameters: List[Variable],
                 verbose: int = 0,
                 engine: Union[EngineLM, str] = None,
                 constraints: List[str] = None,
                 new_variable_tags: List[str] = None,
                 optimizer_system_prompt: str = SAM_OPTIMIZER_SYSTEM_PROMPT,
                 in_context_examples: List[str] = None,
                 gradient_memory: int = 0):

        super().__init__(parameters)

        if new_variable_tags is None:
            self.new_variable_tags = ["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]
        else:
            self.new_variable_tags = new_variable_tags

        self.engine = validate_engine_or_get_default(engine)
        self.verbose = verbose

        self.constraints = constraints if constraints is not None else []
        self.do_constrained = (len(self.constraints) > 0)

        self.in_context_examples = in_context_examples if in_context_examples is not None else []
        self.do_in_context_examples = (len(self.in_context_examples) > 0)

        self.gradient_memory = gradient_memory
        self.gradient_memory_dict = defaultdict(list)
        self.do_gradient_memory = (gradient_memory > 0)
        self.optimizer_system_prompt = optimizer_system_prompt.format(
            new_variable_start_tag=self.new_variable_tags[0],
            new_variable_end_tag=self.new_variable_tags[1]
        )

    @property
    def constraint_text(self):
        constraints_ordered = [f"Constraint {i + 1}: {constraint}" for i, constraint in enumerate(self.constraints)]
        return "\n".join(constraints_ordered)

    def get_gradient_memory_text(self, variable: Variable):
        grad_memory = ""
        variable_grad_memory = self.gradient_memory_dict[variable][-self.gradient_memory:]
        for i, grad_info in enumerate(variable_grad_memory):
            grad_memory += f"\n<FEEDBACK-{i + 1}> {grad_info['value']}</FEEDBACK-{i + 1}>\n"
        return grad_memory

    def update_gradient_memory(self, variable: Variable):
        self.gradient_memory_dict[variable].append({"value": variable.get_gradient_text()})


    def _update_prompt(self, variable: Variable) -> Union[str, List[Union[str, bytes]]]:
        grad_memory = self.get_gradient_memory_text(variable)
        constraint_text = self.constraint_text
        perturbed_value = "PERTURBATION_INFO_NOT_FOUND"
        original_value = "ORIGINAL_INFO_NOT_FOUND"
        if variable.gradients:
            first_gradient = next(iter(variable.gradients))
            grad_context = variable.gradients_context.get(first_gradient, {})
            if "perturbed_variable_value" in grad_context:
                perturbed_value = grad_context["perturbed_variable_value"]
            else:
                logger.warning("Could not find 'perturbed_variable_value' in gradient context.")
            if "original_variable_value" in grad_context:
                original_value = grad_context["original_variable_value"]
            else:
                logger.warning("Could not find 'perturbed_variable_value' in gradient context.")
        optimizer_information = {
            "variable_desc": variable.get_role_description(),
            "original_variable_value": original_value,
            "perturbed_variable_value": perturbed_value,
            "variable_grad": get_gradient_and_context_text(variable),
            "variable_short": variable.get_short_value(),
            "constraint_text": constraint_text,
            "new_variable_start_tag": self.new_variable_tags[0],
            "new_variable_end_tag": self.new_variable_tags[1],
            "in_context_examples": "\n".join(self.in_context_examples),
            "gradient_memory": grad_memory
        }

        prompt = construct_sam_tgd_prompt(
            do_constrained=self.do_constrained,
            do_in_context_examples=self.do_in_context_examples,
            do_gradient_memory=self.do_gradient_memory,
            **optimizer_information
        )
        logger.info(f"LATO prompt for update", extra={"prompt": str(prompt)})
        return prompt

    def step(self):
        for parameter in self.parameters:
            prompt_update_parameter = self._update_prompt(parameter)
            new_text = self.engine(prompt_update_parameter, system_prompt=self.optimizer_system_prompt)

            logger.info(f"LATO optimizer response", extra={"optimizer.response": new_text})
            if isinstance(prompt_update_parameter, list):
                for item in prompt_update_parameter:
                    print(item)
            else:
                print(prompt_update_parameter)
            try:
                new_value = new_text.split(self.new_variable_tags[0])[1].split(self.new_variable_tags[1])[0].strip()
            except IndexError:
                logger.error(f"LATO optimizer response could not be indexed",
                             extra={"optimizer.response": new_text})
                raise IndexError(
                    f"LATO optimizer response could not be indexed. The optimizer model may have failed to follow instructions with the complex TARE prompt. Response: {new_text}")
            parameter.set_value(new_value)
            logger.info(f"LATO updated text", extra={"parameter.value": parameter.value})

            if self.verbose:
                print(parameter.value)

            if self.do_gradient_memory:
                self.update_gradient_memory(parameter)