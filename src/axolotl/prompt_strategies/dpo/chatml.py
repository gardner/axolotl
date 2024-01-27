cat <<EOF > src/axolotl/prompt_strategies/dpo/chatml.py
"""
DPO strategies for chatml
"""

def fix_sample(sample):
    possible_prompt_keys = ['instruction', 'question', 'input']
    possible_chosen_keys = ['chosen_response', 'chosen']
    possible_rejected_keys = ['rejected_response', 'rejected']

    if not "prompt" in sample:
        for key in possible_prompt_keys:
            if key in sample:
                sample["prompt"] = sample[key]
                del sample[key]
                break

    if not "chosen" in sample:
        for key in possible_chosen_keys:
            if key in sample:
                sample["chosen"] = sample[key]
                del sample[key]
                break

    if not "rejected" in sample:
        for key in possible_rejected_keys:
            if key in sample:
                sample["rejected"] = sample[key]
                del sample[key]
                break

    return sample


def flexible(
    cfg,
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        sample = fix_sample(sample)
        ## If the prompt isn't already in chatml format then convert it
        if "<|im_start|>" not in sample["prompt"]:
            if "system" in sample and sample["system"]:
                sample["prompt"] = (
                    f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                    f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
                )
            else:
                sample[
                    "prompt"
                ] = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"

        ## Sometimes the chosen and rejected responses already have the <|im_end|> string
        ## If not then add it:
        if "<|im_end|>" not in sample["chosen"]:
            sample["chosen"] = f"{sample['chosen']}<|im_end|>"

        if "<|im_end|>" not in sample["rejected"]:
            sample["rejected"] = f"{sample['rejected']}<|im_end|>"

        return sample

    return transform_fn



def argilla(
    cfg,
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen_response']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected_response']}<|im_end|>"
        return sample

    return transform_fn


def icr(
    cfg,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    chatml transforms for datasets with system, input, chosen, rejected
    ex. https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected']}<|im_end|>"
        return sample

    return transform_fn


def intel(cfg):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For Intel Orca DPO Pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected']}<|im_end|>"
        return sample

    return transform_fn


def prompt_pairs(cfg):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected']}<|im_end|>"
        return sample

    return transform_fn


def ultra(cfg):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for ultrafeedback binarized conversations
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen'][1]['content']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected'][1]['content']}<|im_end|>"
        return sample

    return transform_fn

EOF