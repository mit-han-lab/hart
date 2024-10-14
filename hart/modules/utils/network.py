import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from hart.train.utils import get_checkpoint_path

__all__ = [
    "load_partially_matched_checkpoint",
    "is_parallel",
    "get_device",
    "get_params_num",
    # "inference_macs",
    "get_same_padding",
    "make_divisible",
    "get_scheduled_size",
    "resize",
]


def load_partially_matched_checkpoint(model: nn.Module, resume_path: str):
    resume_path, _ = get_checkpoint_path(resume_path)
    checkpoint_path = os.path.join(resume_path, "pytorch_model.bin")
    ema_checkpoint_path = os.path.join(resume_path, "ema_model.bin")
    if os.path.exists(ema_checkpoint_path):
        checkpoint_path = ema_checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    new_checkpoint = dict()
    for name, param in model.named_parameters():
        if name in checkpoint:
            if checkpoint[name].shape == param.shape:
                new_checkpoint[name] = checkpoint[name]
        else:
            print(name)
            continue
            # raise NotImplementedError
    model.load_state_dict(new_checkpoint, strict=False)


def is_parallel(model: nn.Module) -> bool:
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device


def get_params_num(model: nn.Module, unit: float = 1e6, train_only=True) -> float:
    n_params = 0
    for p in model.parameters():
        if train_only and not p.requires_grad:
            continue
        n_params += p.numel()
    return n_params / unit


def get_same_padding(
    kernel_size: Union[int, Tuple[int, ...]]
) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def make_divisible(
    v: float, divisor: Optional[int], min_val=None, ceil=False, round_down=False
) -> int:
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :param ceil:
    :param round_down:
    :return:
    """
    if divisor is None:
        return int(v)

    if ceil:
        return math.ceil(v / divisor) * divisor
    elif round_down:
        return int(v / divisor) * divisor
    else:
        new_v = max(int(min_val or divisor), int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v


def get_scheduled_size(
    base_size: int, min_ratio: float, max_ratio: float, progress: float, divisor: int
) -> int:
    min_size = make_divisible(base_size * min_ratio, divisor, ceil=True)
    max_size = make_divisible(base_size * max_ratio, divisor, round_down=True)
    all_size = list(range(min_size, max_size + divisor, divisor))
    idx = int(len(all_size) * progress)
    return all_size[idx]


def resize(
    x: torch.Tensor,
    size: Optional[any] = None,
    scale_factor: Optional[List[float]] = None,
    mode: str = "bicubic",
    align_corners: bool = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")
