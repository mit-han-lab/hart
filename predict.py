# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import subprocess
import time
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "0"

original_dir = os.getcwd()
try:
    subprocess.run(["python", "setup.py", "install"], cwd="hart/kernels")
    import pkg_resources

    dist = pkg_resources.get_distribution("hart_backend")
    egg_path = dist.location
    sys.path.append(egg_path)
finally:
    os.chdir(original_dir)

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from cog import BasePredictor, Input, Path

from hart.modules.models.transformer.hart_transformer_t2i import (
    HARTForT2IConfig,
    HARTForT2I,
)
from hart.utils import encode_prompts, llm_system_prompt, safety_check

AutoConfig.register("hart_transformer_t2i", HARTForT2IConfig)
AutoModel.register(HARTForT2IConfig, HARTForT2I)


# cache files from mit-han-lab/Qwen2-VL-1.5B-Instruct, mit-han-lab/hart-0.7b-1024px, and google/shieldgemma-2b
MODEL_CACHE = "model_cache"
MODEL_URL = (
    f"https://weights.replicate.delivery/default/mit-han-lab/hart/{MODEL_CACHE}.tar"
)

os.environ.update(
    {
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": MODEL_CACHE,
        "TORCH_HOME": MODEL_CACHE,
        "HF_DATASETS_CACHE": MODEL_CACHE,
        "TRANSFORMERS_CACHE": MODEL_CACHE,
        "HUGGINGFACE_HUB_CACHE": MODEL_CACHE,
    }
)


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        model_path = f"{MODEL_CACHE}/mit-han-lab/hart-0.7b-1024px/llm"
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda")
        self.model.eval()
        # use_ema by default
        self.model.load_state_dict(
            torch.load(os.path.join(model_path, "ema_model.bin"))
        )

        text_model_path = f"{MODEL_CACHE}/mit-han-lab/Qwen2-VL-1.5B-Instruct"
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = AutoModel.from_pretrained(
            text_model_path, torch_dtype=torch.float16
        ).to("cuda")
        self.text_model.eval()

        shield_model_path = f"{MODEL_CACHE}/google/shieldgemma-2b"
        self.safety_checker_tokenizer = AutoTokenizer.from_pretrained(shield_model_path)
        self.safety_checker_model = AutoModelForCausalLM.from_pretrained(
            shield_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.safety_checker_model.eval()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a horse on the moon, oil painting by Van Gogh.",
        ),
        max_token_length: int = Input(default=300),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=4.5
        ),
        more_smooth: bool = Input(
            description="Turn on for more visually smooth samples", default=True
        ),
        use_llm_system_prompt: bool = Input(default=True),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        assert not safety_check.is_dangerous(
            self.safety_checker_tokenizer, self.safety_checker_model, prompt
        ), f"The prompt id not pass the safety checker, please use a different prompt."

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator().manual_seed(seed)

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
                    [prompt],
                    self.text_model,
                    self.text_tokenizer,
                    max_token_length,
                    llm_system_prompt,
                    use_llm_system_prompt,
                )

                infer_func = self.model.autoregressive_infer_cfg

                output_imgs = infer_func(
                    B=context_tensor.size(0),
                    label_B=context_tensor,
                    cfg=guidance_scale,
                    g_seed=seed,
                    more_smooth=more_smooth,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )

        sample_imgs_np = output_imgs.clone().mul_(255).cpu().numpy()
        cur_img = sample_imgs_np[0]
        cur_img = cur_img.transpose(1, 2, 0).astype(np.uint8)
        cur_img_store = Image.fromarray(cur_img)

        out_path = "/tmp/out.png"
        cur_img_store.save(out_path)
        return Path(out_path)
