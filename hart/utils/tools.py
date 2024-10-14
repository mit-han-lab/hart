import json
from functools import lru_cache
from typing import Dict, Sequence

import torch
import transformers

__rank, __local_rank, __world_size, __device = (
    0,
    0,
    1,
    "cuda" if torch.cuda.is_available() else "cpu",
)


# Modified from VILA
def tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    padding_mode: str = "longest",
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding_mode,
            max_length=max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    return input_ids


def encode_prompts(
    prompts,
    text_model,
    text_tokenizer,
    text_tokenizer_max_length,
    system_prompt=None,
    use_llm_system_prompt=False,
):
    device = text_model.device
    tokenized_prompts = tokenize_fn(
        prompts,
        tokenizer=text_tokenizer,
        max_length=text_tokenizer_max_length,
        padding_mode="max_length",
    )
    context_tokens = torch.stack(tokenized_prompts).to(device)
    context_mask = context_tokens != text_tokenizer.pad_token_id
    context_position_ids = torch.cumsum(context_mask, dim=1) - 1

    if not use_llm_system_prompt:
        context_tensor = text_model(
            context_tokens, attention_mask=context_mask, output_hidden_states=True
        ).hidden_states[-1]
    else:
        system_prompt_tokens = tokenize_fn(
            [system_prompt],
            tokenizer=text_tokenizer,
            max_length=text_tokenizer_max_length,
            padding_mode="longest",
        )
        system_prompt_tokens = system_prompt_tokens[0].to(context_tokens.device)
        system_prompt_tokens = system_prompt_tokens.unsqueeze(0)
        system_prompt_tokens = system_prompt_tokens.repeat(context_tokens.shape[0], 1)
        system_prompt_mask = torch.ones_like(context_mask)[
            :, : system_prompt_tokens.shape[1]
        ]
        # include system prompt when calculating embeddings
        # but only keep the embedding for original tokens
        context_tensor = text_model(
            torch.cat([system_prompt_tokens, context_tokens], dim=1),
            attention_mask=torch.cat(
                [
                    system_prompt_mask,
                    context_mask,
                ],
                dim=1,
            ),
            output_hidden_states=True,
        ).hidden_states[-1][:, system_prompt_tokens.shape[1] :]
    context_tensor = context_tensor.float()

    return (context_tokens, context_mask, context_position_ids, context_tensor)


@lru_cache(maxsize=16)
def lru_json_load(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def get_device():
    return __device
