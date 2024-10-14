import torch
from torch.nn.functional import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer

__all__ = ["is_dangerous"]

safety_policy = """
    The prompt must not seek to generate harmful, abusive, hateful, sexually explicit, or otherwise inappropriate content targeting individuals or protected groups.
"""


def is_dangerous(tokenizer, model, user_prompt, threshold=0.5):
    prompt = f"""You are a policy expert trying to help determine whether a user
    prompt is in violation of the defined safety policies.

    <start_of_turn>
    Human Question: {user_prompt.strip()}
    <end_of_turn>

    Our safety principle is defined in the below:

    {safety_policy.strip()}

    Does the human question violate the above principle? Your answer must start
    with 'Yes' or 'No'. And then walk through step by step to be sure we answer
    correctly.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits

    # Extract the logits for the Yes and No tokens
    vocab = tokenizer.get_vocab()
    selected_logits = logits[0, -1, [vocab["Yes"], vocab["No"]]]

    # Convert these logits to a probability with softmax
    probabilities = softmax(selected_logits, dim=0)

    # Return probability of 'Yes'
    score = probabilities[0].item()

    return score > threshold
