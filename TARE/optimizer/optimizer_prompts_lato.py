GLOSSARY_TEXT = """
### Glossary of tags that will be sent to you:
# - <ORIGINAL_VARIABLE>: The original variable that you need to improve.
# - <PERTURBED_VARIABLE>: A slightly perturbed version of the original variable that resulted in the feedback.
# - <ORIGINAL_VARIABLE_LOSS>: The performance of the original variable on the current batch.
# - <PERTURBED_VARIABLE_LOSS>: The performance of the perturbed variable on the current batch.
# - <LM_SYSTEM_PROMPT>: The system prompt for the language model.
# - <LM_INPUT>: The input to the language model.
# - <LM_OUTPUT>: The output of the language model.
# - <FEEDBACK>: The feedback to the variable.
# - <CONVERSATION>: The conversation history.
# - <FOCUS>: The focus of the optimization.
# - <ROLE>: The role description of the variable."""

OPTIMIZER_SYSTEM_PROMPT = (
    "You are an expert optimizer and a creative critic within an advanced AI system. You will be asked to creatively and critically improve text-based variables (prompts, solutions, code, etc.) to make them more effective and robust.\n\n"

    "THE PROCESS:\n"
    "To do this, you will be given an <ORIGINAL_VARIABLE>. This variable was perturbed into a <PERTURBED_VARIABLE>, and the system's performance using this perturbed version resulted in critical <FEEDBACK>.\n\n"

    "YOUR TASK & OBJECTIVES:\n"
    "Based on all available information, your goal is to generate a new, improved version of the **ORIGINAL_VARIABLE**. The new version must achieve the following objectives:\n"
    "1. **Address the Failure**: It must resolve the specific issues pointed out in the provided <FEEDBACK>.\n"
    "2. **Preserve Performance**: It must maintain or improve upon the original variable's good performance.\n"
    "3. **Enhance Robustness**: It must be more resilient to similar small perturbations in the future.\n\n"

    "GUIDING PRINCIPLES:\n"
    "- **Preserve Core Meaning**: Whatever the edit, you must strictly preserve the core task intent and local coherence of the original text."
    "- **Analyze Noisy Feedback**: The provided <FEEDBACK> may be noisy. Critically evaluate it to identify what is important and correct.\n"
    "- **Consider Full Context**: Always pay close attention to the variable's <ROLE> and the full context in which it is used to ensure your improvements are relevant.\n\n"

    "IMPORTANT - OUTPUT FORMAT:\n"
    "You MUST give your response by sending the improved variable between {new_variable_start_tag}{{improved variable}}{new_variable_end_tag} tags. "
    "The text you send between the tags will directly replace the variable.\n\n"
    f"{GLOSSARY_TEXT}"
)

TGD_PROMPT_PREFIX = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The ORIGINAL variable that we are optimizing is:\n<ORIGINAL_VARIABLE>\n{original_variable_value}\n</ORIGINAL_VARIABLE>\n\n"
    "When this variable was slightly perturbed into the following version:\n<PERTURBED_VARIABLE>\n{perturbed_variable_value}\n</PERTURBED_VARIABLE>\n\n"
    "The system received the following feedback based on the PERTURBED version's performance:\n"
    "<CONTEXT>{variable_grad}</CONTEXT>\n\n"
    "Based on the feedback from the perturbed version, improve the ORIGINAL variable ({variable_desc}) to make it more robust.\n"
)

TGD_MULTIPART_PROMPT_INIT = (
    "Here is the role of the variable you will improve: <ROLE>{variable_desc}</ROLE>.\n\n"
    "The ORIGINAL variable that we are optimizing is:\n<ORIGINAL_VARIABLE>\n{original_variable_value}\n</ORIGINAL_VARIABLE>\n\n"
    "When this variable was slightly perturbed into the following version:\n<PERTURBED_VARIABLE>\n{perturbed_variable_value}\n</PERTURBED_VARIABLE>\n\n"
    "The system received the following feedback based on the PERTURBED version's performance:\n"
)


TGD_MULTIPART_PROMPT_PREFIX = (
    "Based on the feedback from the perturbed version, improve the ORIGINAL variable ({variable_desc}).\n"
)


TGD_PROMPT_SUFFIX = (
    "Send the improved variable "
    "in the following format:\n\n{new_variable_start_tag}{{the improved variable}}{new_variable_end_tag}\n\n"
    "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else."
)


MOMENTUM_PROMPT_ADDITION = (
    "Here are the past iterations of this variable:\n\n"
    "<PAST_ITERATIONS>{past_values}</PAST_ITERATIONS>\n\n"
    "Similar feedbacks across different steps suggests that the modifications to the variable are insufficient."
    "If this is the case, please make more significant changes to the variable.\n\n"
)


CONSTRAINT_PROMPT_ADDITION = (
    "You must follow the following constraints:\n\n"
    "<CONSTRAINTS>{constraint_text}</CONSTRAINTS>\n\n"
)


IN_CONTEXT_EXAMPLE_PROMPT_ADDITION = (
    "You must base on the following examples when modifying the {variable_desc}:\n\n"
    "<EXAMPLES>{in_context_examples}</EXAMPLES>\n\n"
)



def construct_tgd_prompt(do_momentum: bool = False,
                         do_constrained: bool = False,
                         do_in_context_examples: bool = False,
                         **optimizer_kwargs):

    if isinstance(optimizer_kwargs["variable_grad"], str):
        multipart = False
        prompt = TGD_PROMPT_PREFIX.format(**optimizer_kwargs)

    else:
        gradient_context = optimizer_kwargs["variable_grad"]
        gradient_context = [TGD_MULTIPART_PROMPT_INIT.format(**optimizer_kwargs)] + gradient_context
        multipart = True
        prompt = TGD_MULTIPART_PROMPT_PREFIX.format(**optimizer_kwargs)

    if do_momentum:
        prompt += MOMENTUM_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_constrained:
        prompt += CONSTRAINT_PROMPT_ADDITION.format(**optimizer_kwargs)

    if do_in_context_examples:
        prompt += IN_CONTEXT_EXAMPLE_PROMPT_ADDITION.format(**optimizer_kwargs)

    prompt += TGD_PROMPT_SUFFIX.format(**optimizer_kwargs)

    if not multipart:
        return prompt

    else:
        return gradient_context + [prompt]


GRADIENT_TEMPLATE = (
    "Here is a conversation:\n\n<CONVERSATION>{context}</CONVERSATION>\n\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)
GRADIENT_MULTIPART_TEMPLATE = (
    "Above is a conversation with a language model.\n"
    "This conversation is potentially part of a larger system. The output is used as {response_desc}\n\n"
    "Here is the feedback we got for {variable_desc} in the conversation:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)