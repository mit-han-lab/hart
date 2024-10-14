import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as tdist
from torch.nn import functional as F

from hart.modules.models.autoencoder.quantize.var_quantize_multiple_res import (
    VectorQuantizer2 as VARQuantizer,
)

__all__ = ["HARTHybridQuantizer"]


class HARTHybridQuantizer(VARQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        f_BChw: torch.Tensor,
        patch_nums=None,
        ret_usages=True,
        skip_continuous_prob=0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[float], torch.Tensor, List[torch.Tensor]]:
        dtype = f_BChw.dtype
        if dtype != torch.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        if patch_nums is None:
            patch_nums = self.v_patch_nums

        idx_list = []

        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(
                self.vocab_size, dtype=torch.float, device=f_BChw.device
            )
            SN = len(patch_nums)
            for si, pn in enumerate(patch_nums):  # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                        if (si != SN - 1)
                        else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                    )
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(
                        rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0),
                        dim=1,
                    )
                else:
                    rest_NC = (
                        F.interpolate(f_rest, size=(pn, pn), mode="area")
                        .permute(0, 2, 3, 1)
                        .reshape(-1, C)
                    )
                    d_no_grad = torch.sum(
                        rest_NC.square(), dim=1, keepdim=True
                    ) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False
                    )
                    d_no_grad.addmm_(
                        rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                    )  # (B*h*w, vocab_size)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()
                if self.training:
                    handler = tdist.all_reduce(hit_V, async_op=True)

                # calc loss
                idx_list.append(idx_N)
                idx_Bhw = idx_N.view(B, pn, pn)
                h_BChw = F.interpolate(
                    self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                    size=(H, W),
                    mode="bicubic",
                ).contiguous()
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw

                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul_(
                    self.beta
                ) + F.mse_loss(f_hat, f_no_grad)

            # optionally decode the continuous latent
            p = random.random()
            if p >= skip_continuous_prob:
                # skip the final stage with 50% probability
                h_BChw = f_rest.clone()
                f_hat = f_hat + h_BChw
                f_rest -= h_BChw

            mean_vq_loss *= 1.0 / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BChw)

        margin = 1
        if ret_usages:
            usages = (vocab_hit_V >= margin).float().mean().item() * 100
        else:
            usages = None
        return f_hat, usages, mean_vq_loss, idx_list

    def f_to_idxBl_and_frest(
        self,
        f_BChw: torch.Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        **kwargs,
    ) -> List[Union[torch.Tensor, torch.LongTensor]]:
        # return: [idx_Bl for si in [0, SN - 2], h_BChw for SN - 1]
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        idx_Bl_and_frest: List[torch.Tensor] = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1])
            for pn in (v_patch_nums or self.v_patch_nums)
        ]
        assert (
            patch_hws[-1][0] == H and patch_hws[-1][1] == W
        ), f"{patch_hws[-1]=} != ({H=}, {W=})"

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):
            z_NC = (
                F.interpolate(f_rest, size=(ph, pw), mode="area")
                .permute(0, 2, 3, 1)
                .reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )
            d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                self.embedding.weight.data.square(), dim=1, keepdim=False
            )
            d_no_grad.addmm_(
                z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
            )  # (B*h*w, vocab_size)
            idx_N = torch.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view(B, ph, pw)
            h_BChw = F.interpolate(
                self.embedding(idx_Bhw).permute(0, 3, 1, 2),
                size=(H, W),
                mode="bicubic",
            ).contiguous()
            if si != SN - 1:
                # consistency
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            if si == SN - 1:
                f_rest_wo_last_discrete = f_rest.clone()
            f_rest.sub_(h_BChw)
            idx_Bl_and_frest.append(idx_N.reshape(B, ph * pw))

        h_BChw = f_rest
        f_hat = f_hat + h_BChw
        idx_Bl_and_frest.append(
            f_rest_wo_last_discrete.clone()
            .permute(0, 2, 3, 1)
            .reshape(-1, ph * pw, self.Cvae)
        )

        return idx_Bl_and_frest

    def get_next_autoregressive_input(
        self,
        si: int,
        SN: int,
        f_hat: torch.Tensor,
        h_BChw: torch.Tensor,
        patch_nums=None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:  # only used in VAR inference
        if patch_nums is None:
            patch_nums = self.v_patch_nums
        HW = patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )  # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(
                f_hat,
                size=(patch_nums[si + 1], patch_nums[si + 1]),
                mode="area",
            )
        else:
            h = h_BChw
            f_hat.add_(h)
            return f_hat, f_hat
