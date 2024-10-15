#!/usr/bin/env python

import argparse
import copy
import os
import random
import uuid

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from hart.modules.models.transformer import HARTForT2I
from hart.utils import default_prompts, encode_prompts, llm_system_prompt, safety_check

DESCRIPTION = (
    """# HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"""
    + """\n[\\[Paper\\]](https://arxiv.org/abs/2410.10812) [\\[Project\\]](https://hanlab.mit.edu/projects/hart) [\\[GitHub\\]](https://github.com/mit-han-lab/hart)"""
    + """\n<p>Note: We will replace unsafe prompts with a default prompt: \"A red heart.\"</p>"""
)
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo may not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_IMAGES_PER_PROMPT = 1


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


@spaces.GPU(enable_queue=True)
def generate(
    prompt: str,
    seed: int = 0,
    # width: int = 1024,
    # height: int = 1024,
    guidance_scale: float = 4.5,
    randomize_seed: bool = False,
    progress=gr.Progress(track_tqdm=True),
):
    global text_model, text_tokenizer
    # pipe.to(device)
    seed = int(randomize_seed_fn(seed, randomize_seed))
    generator = torch.Generator().manual_seed(seed)

    if safety_check.is_dangerous(
        safety_checker_tokenizer, safety_checker_model, prompt
    ):
        prompt = "A red heart."

    prompts = [prompt]

    with torch.inference_mode():
        with torch.autocast(
            "cuda", enabled=True, dtype=torch.float16, cache_enabled=True
        ):

            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tensor,
            ) = encode_prompts(
                prompts,
                text_model,
                text_tokenizer,
                args.max_token_length,
                llm_system_prompt,
                args.use_llm_system_prompt,
            )

            infer_func = model.autoregressive_infer_cfg

            output_imgs = infer_func(
                B=context_tensor.size(0),
                label_B=context_tensor,
                cfg=args.cfg,
                g_seed=seed,
                more_smooth=args.more_smooth,
                context_position_ids=context_position_ids,
                context_mask=context_mask,
            )

    # bs, 3, r, r
    images = []
    sample_imgs_np = output_imgs.clone().mul_(255).cpu().numpy()
    num_imgs = sample_imgs_np.shape[0]
    for img_idx in range(num_imgs):
        cur_img = sample_imgs_np[img_idx]
        cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
        cur_img_store = Image.fromarray(cur_img)
        images.append(cur_img_store)

    return images, seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="The path to HART model.",
        default="pretrained_models/HART-1024",
    )
    parser.add_argument(
        "--text_model_path",
        type=str,
        help="The path to text model, we employ Qwen2-VL-1.5B-Instruct by default.",
        default="Qwen2-VL-1.5B-Instruct",
    )
    parser.add_argument(
        "--shield_model_path",
        type=str,
        help="The path to shield model, we employ ShieldGemma-2B by default.",
        default="google/shieldgemma-2b",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--max_token_length", type=int, default=300)
    parser.add_argument("--use_llm_system_prompt", type=bool, default=True)
    parser.add_argument(
        "--cfg", type=float, help="Classifier-free guidance scale.", default=4.5
    )
    parser.add_argument(
        "--more_smooth",
        type=bool,
        help="Turn on for more visually smooth samples.",
        default=True,
    )
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()

    if args.use_ema:
        model.load_state_dict(
            torch.load(os.path.join(args.model_path, "ema_model.bin"))
        )

    text_tokenizer = AutoTokenizer.from_pretrained(args.text_model_path)
    text_model = AutoModel.from_pretrained(
        args.text_model_path, torch_dtype=torch.float16
    ).to(device)
    text_model.eval()
    text_tokenizer_max_length = args.max_token_length

    safety_checker_tokenizer = AutoTokenizer.from_pretrained(args.shield_model_path)
    safety_checker_model = AutoModelForCausalLM.from_pretrained(
        args.shield_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).to(device)

    examples = [
        "melting apple",
        "neon holography crystal cat",
        "A dog that has been meditating all the time",
        "An astronaut riding a horse on the moon, oil painting by Van Gogh.",
        "8k uhd A man looks up at the starry sky, lonely and ethereal, Minimalism, Chaotic composition Op Art",
        "Full body shot, a French woman, Photography, French Streets background, backlighting, rim light, Fujifilm.",
        "Steampunk makeup, in the style of vray tracing, colorful impasto, uhd image, indonesian art, fine feather details with bright red and yellow and green and pink and orange colours, intricate patterns and details, dark cyan and amber makeup. Rich colourful plumes. Victorian style.",
    ]

    css = """
    .gradio-container{max-width: 560px !important}
    h1{text-align:center}
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown(DESCRIPTION)
        gr.DuplicateButton(
            value="Duplicate Space for private use",
            elem_id="duplicate-button",
            visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
        )
        with gr.Group():
            with gr.Row():
                prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                    container=False,
                )
                run_button = gr.Button("Run", scale=0)

            result = gr.Gallery(
                label="Result",
                columns=NUM_IMAGES_PER_PROMPT,
                show_label=False,
                # height=800,
            )
            with gr.Accordion("Advanced options", open=False):
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=args.seed,
                )
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                with gr.Row():
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.1,
                        maximum=20,
                        step=0.1,
                        value=4.5,
                    )

        gr.Examples(
            examples=examples,
            inputs=prompt,
            outputs=[result, seed],
            fn=generate,
            cache_examples=CACHE_EXAMPLES,
        )

        gr.on(
            triggers=[
                prompt.submit,
                run_button.click,
            ],
            fn=generate,
            inputs=[
                prompt,
                seed,
                guidance_scale,
                randomize_seed,
            ],
            outputs=[result, seed],
            api_name="run",
        )

    demo.queue(max_size=20).launch(share=True)
