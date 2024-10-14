"""This file contains code for text-to-image HART Transformer.

This file is adopted and modified from https://github.com/FoundationVision/VAR/blob/main/models/var.py
"""

import math
import os
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, PreTrainedModel

from hart.modules.diffusion.diffloss import DiffLoss
from hart.modules.models.autoencoder import (
    HARTAutoEncoder,
    HARTAutoEncoderWithDisc,
    HARTHybridQuantizer,
)
from hart.modules.models.transformer.configuration import HARTForT2IConfig
from hart.modules.networks.basic_hart import (
    AdaLNBeforeHead,
    AdaLNSelfAttn,
    LlamaRMSNormFused,
    TimestepEmbedder,
)
from hart.modules.networks.utils import (
    gumbel_softmax_with_rng,
    sample_with_top_k_top_p_,
)
from hart.utils import get_device


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(
        masking,
        dim=-1,
        index=order[:, : mask_len.long()],
        src=torch.ones(bsz, seq_len).cuda(),
    ).bool()
    return masking


class CopyableGenerator(torch.Generator):
    def __deepcopy__(self, memo):
        new_generator = CopyableGenerator(device=self.device)
        new_generator.set_state(self.get_state())
        return new_generator


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)  # B16C


class HARTForT2I(PreTrainedModel):
    config_class = HARTForT2IConfig

    def __init__(self, config: HARTForT2IConfig):
        super().__init__(config)
        self.supports_gradient_checkpointing = True
        # 0. hyperparameters

        embed_dim = config.embed_dim
        num_heads = config.num_heads
        vae_path = config.vae_path
        if vae_path is None:
            vae_path = os.path.join(
                os.path.dirname(config._name_or_path.rstrip("/")), "tokenizer"
            )
        depth = config.depth
        drop_rate = config.drop_rate
        cond_drop_rate = config.cond_drop_rate
        drop_path_rate = config.drop_path_rate
        attn_drop_rate = config.attn_drop_rate
        mlp_ratio = config.mlp_ratio
        norm_eps = config.norm_eps
        shared_aln = config.shared_aln
        attn_l2_norm = config.attn_l2_norm
        context_token = config.context_token
        context_dim = config.context_dim
        patch_nums = config.patch_nums
        flash_if_available, fused_if_available = (
            config.flash_if_available,
            config.fused_if_available,
        )
        self.mlp_type = mlp_type = config.mlp_type
        self.attn_type = attn_type = config.attn_type
        if self.attn_type == "gpt2":
            norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        elif self.attn_type == "llama":
            norm_layer = partial(LlamaRMSNormFused, eps=norm_eps)
        else:
            raise NotImplementedError
        self.disable_aln = config.disable_aln
        self.use_timestep_embed = use_timestep_embed = config.use_timestep_embed
        self.sep_aln_pooling_mode = sep_aln_pooling_mode = config.sep_aln_pooling_mode
        self.use_cross_attn = use_cross_attn = config.use_cross_attn

        self.diffusion_head_repeats = diffusion_head_repeats = (
            config.diffusion_head_repeats
        )
        # MAR variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25
        mask_ratio_min = 0.5
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        vae_local = HARTAutoEncoderWithDisc.from_pretrained(vae_path).vae
        vae_local = vae_local.cuda()
        vae_local.requires_grad_(False)

        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = tuple(patch_nums)
        self.L = sum(pn**2 for pn in self.patch_nums)
        # self.first_l = self.patch_nums[0] ** 2
        self.first_l = context_token
        self.begin_ends = []
        self.begin_ends.append((0, context_token))
        cur = context_token
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = CopyableGenerator(device=get_device())

        # 1. input (word) embedding
        quant: HARTHybridQuantizer = vae_local.quantize
        self.vae_proxy: Tuple[HARTAutoEncoder] = (vae_local,)
        self.vae_quant_proxy: Tuple[HARTHybridQuantizer] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        # self.num_classes = num_classes
        # self.uniform_prob = torch.full(
        #     (1, num_classes),
        #     fill_value=1.0 / num_classes,
        #     dtype=torch.float32,
        #     device=get_device(),
        # )
        self.context_token = context_token
        self.context_dim = context_dim
        self.context_shape = (context_token, context_dim)
        self.context_embed = nn.Linear(context_dim, self.D)
        if config.use_context_norm:
            self.context_norm = norm_layer(context_dim, scale=config.context_norm_scale)
        else:
            self.context_norm = nn.Identity()

        nn.init.trunc_normal_(self.context_embed.weight.data, mean=0, std=init_std)
        if attn_type == "gpt2" or self.context_token == 0:
            # gpt2 uses absolute pos emb for context tokens
            # c2i also adds this absolute pos emb
            self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
            nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        else:
            self.pos_start = None

        # 3. absolute position embedding
        self.last_level_pns = self.patch_nums[-1] ** 2
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            if i > 0:
                pe = torch.empty(1, pn * pn, self.C)
            else:
                pe = torch.empty(1, context_token, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L + context_token - 1, self.C)
        if self.attn_type == "gpt2":
            self.pos_1LC = nn.Parameter(pos_1LC)
        elif self.attn_type == "llama":
            self.pos_1LC = None
        else:
            raise NotImplementedError

        if not self.use_timestep_embed:
            # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
            self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
            nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        else:
            self.lvl_embed = TimestepEmbedder(embed_dim)

        # 4. backbone blocks
        self.shared_ada_lin = nn.Identity()

        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                    mlp_type=mlp_type,
                    attn_type=attn_type,
                    max_position_embeddings=2
                    ** int(math.ceil(math.log2(self.L + context_token - 1))),
                    patch_nums=self.patch_nums,
                    context_token=self.context_token,
                    disable_aln=self.disable_aln,
                    sep_aln_pooling_mode=self.sep_aln_pooling_mode,
                    use_cross_attn=self.use_cross_attn,
                )
                for block_idx in range(depth)
            ]
        )

        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat(
            [torch.full((context_token,), 0)]
            + [
                torch.full((pn * pn,), i + 1)
                for i, pn in enumerate(self.patch_nums[1:])
            ]
        ).view(1, self.L + context_token - 1, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
            1, 1, self.L + context_token - 1, self.L + context_token - 1
        )
        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )
        print(attn_bias_for_masking.shape)

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.decoder_norm = norm_layer(self.C)
        # self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.last_level_pns, self.C))

        self.diffloss = DiffLoss(
            target_channels=self.Cvae,
            z_channels=self.C,
            width=config.diff_width,
            depth=config.diff_depth,
            num_sampling_steps=config.num_sampling_steps,
            sampler=config.sampler,
        )
        self.diffusion_batch_mul = config.diffusion_batch_mul

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def forward_diff_loss(self, z, target, mask=None):
        bs, seq_len, _ = target.shape
        target = target.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        context_position_ids: torch.Tensor = None,
        context_mask: torch.Tensor = None,
        final_stage=0,
        num_maskgit_iters=1,
    ) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        # num_maskgit_iters = 1
        # final_stage = 2
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        assert label_B is not None
        assert label_B.shape[1] == self.context_token

        sos = cond_BD = self.context_embed(
            self.context_norm(
                torch.cat((label_B, torch.full_like(label_B, fill_value=0.0)), dim=0)
            )
        )
        # Haotian: need to handle CFG here so we replicate context position ids
        context_position_ids = torch.cat(
            (context_position_ids, torch.full_like(context_position_ids, fill_value=0)),
            dim=0,
        )

        b = context_mask.shape[0]
        context_mask = torch.cat(
            (context_mask, torch.full_like(context_mask, fill_value=0))
        )
        context_mask[b:, 0] = 1

        if self.pos_1LC is not None:
            lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        else:
            lvl_pos = self.lvl_embed(self.lvl_1L)

        if self.pos_start is not None:
            next_token_map = (
                sos.expand(2 * B, self.first_l, -1)
                + self.pos_start.expand(2 * B, self.first_l, -1)
                + lvl_pos[:, : self.first_l]
            )
        else:
            next_token_map = (
                sos.expand(2 * B, self.first_l, -1) + lvl_pos[:, : self.first_l]
            )

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for b in self.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums[:-1]):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            if si > 0:
                cur_L += pn * pn
            else:
                cur_L += self.context_token
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                # Haotian: si used for position embed
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=si,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            if si == self.num_stages_minus_1:
                last_layer_cond = x

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            # Haotian: Added for text-conditioned generation
            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)

            f_hat, next_token_map = self.vae_quant_proxy[
                0
            ].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw, patch_nums=self.patch_nums
            )

            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = (
                self.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(
                2, 1, 1
            )  # double the batch sizes due to CFG

        ################ last stage maskgit ################
        si = len(self.patch_nums) - 1
        mask = torch.ones(B, self.last_level_pns).cuda()
        tokens = torch.zeros(B, self.last_level_pns, self.Cvae).cuda()
        orders = self.sample_orders(B)

        num_iter = num_maskgit_iters
        indices = list(range(num_iter))
        # generate latents with maskgit
        for step in indices:
            # mask_ratio = 1 - (step + 1) / num_iter
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.last_level_pns * mask_ratio)]).cuda()
            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )
            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, B, self.last_level_pns)
            if step >= num_iter - 1:
                mask_to_pred = mask[:B].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:B].bool(), mask_next.bool())
            mask = mask_next
            cur_mask = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            cur_mask = cur_mask.nonzero(as_tuple=True)
            x = next_token_map[cur_mask].reshape(2 * B, -1, self.C)
            for b in self.blocks:
                # Haotian: si used for position embed
                # note: m_maskgit makes sure that PEs are correct.
                x = b(
                    x=x,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=None,
                    si=len(self.patch_nums) - 1,
                    m_maskgit=cur_mask,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
            logits_BlV = self.get_logits(x, cond_BD)
            last_layer_cond = x
            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            si = len(self.patch_nums) - 1
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=(600 if si < 7 else 300),
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            if final_stage == 0:
                # sample with diffusion model
                last_stage_discrete_cond = self.vae_quant_proxy[0].embedding(idx_Bl)
                last_stage_discrete_cond = self.word_embed(last_stage_discrete_cond)
                last_stage_discrete_cond = torch.cat(
                    [last_stage_discrete_cond, last_stage_discrete_cond], dim=0
                )
                last_stage_cond = self.decoder_norm(
                    last_layer_cond + last_stage_discrete_cond
                )
                bs, cur_seq_len, _ = last_stage_cond.shape
                ##### begin baseline sampling #####
                last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
                h_BChw_diff = self.diffloss.sample(
                    z=last_stage_cond, temperature=1.0, cfg=t
                )
                ##### end baseline sampling #####
                h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
                # [B, L, Cvae]
                h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
                # update feature map
                tokens[mask_to_pred] = (h_BChw + h_BChw_diff).reshape(-1, self.Cvae)
            else:
                tokens[mask_to_pred] = h_BChw.reshape(-1, self.Cvae)
        h_BChw_final = tokens.transpose(1, 2).reshape(
            B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]
        )
        f_hat += h_BChw_final

        ################ last stage maskgit ################

        for b in self.blocks:
            b.attn.kv_caching(False)
        return (
            self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        )  # de-normalize, from [-1, 1] to [0, 1]

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.last_level_pns)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        # we cannot mask out all the tokens
        num_masked_tokens = min(int(np.ceil(seq_len * mask_rate)), seq_len - 32)
        mask = torch.zeros(bsz, seq_len, device=x.device)
        # all first few stages are kept
        mask_keep = torch.zeros(
            bsz, self.L - seq_len + self.context_token - 1, device=x.device
        )
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        mask_full = torch.cat([mask_keep, mask], dim=1).contiguous()
        return mask_full, mask

    def forward(
        self,
        context: torch.Tensor,
        x_BLCv_wo_first_l: torch.Tensor,
        context_position_ids: torch.Tensor,
        context_mask: torch.Tensor,
        last_layer_gt: torch.Tensor = None,
        last_layer_gt_discrete: torch.Tensor = None,
    ) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = (
            self.begin_ends[self.prog_si]
            if self.prog_si >= 0
            else (0, self.L + self.context_token - 1)
        )
        B = x_BLCv_wo_first_l.shape[0]
        orders = self.sample_orders(bsz=B)
        mask, mask_wo_prev_stages = self.random_masking(
            x_BLCv_wo_first_l[:, -self.last_level_pns :, :], orders
        )
        mask_for_attn = (1 - mask)[:, self.context_token :].nonzero(as_tuple=True)
        mask = (1 - mask).nonzero(as_tuple=True)
        mask_wo_prev_stages = (1 - mask_wo_prev_stages).nonzero(as_tuple=True)
        last_layer_gt = last_layer_gt[mask_wo_prev_stages].reshape(
            B, -1, last_layer_gt.shape[-1]
        )
        last_layer_gt_discrete = last_layer_gt_discrete[mask_wo_prev_stages].reshape(
            B, -1
        )
        ed = (
            last_layer_gt.shape[1]
            + self.L
            + self.context_token
            - 1
            - self.last_level_pns
        )

        with torch.cuda.amp.autocast(enabled=False):
            drop_pos = torch.where(
                torch.randn(B, device=context.device) < self.cond_drop_rate
            )[0]
            context[drop_pos] *= 0

            sos = cond_BD = self.context_embed(self.context_norm(context))
            if self.pos_start is not None:
                sos = sos.expand(B, self.first_l, -1) + self.pos_start.expand(
                    B, self.first_l, -1
                )
            else:
                sos = sos.expand(B, self.first_l, -1)

            if self.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat(
                    (sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1
                )

            # apply maskgit
            x_BLC = x_BLC[mask].reshape(B, -1, x_BLC.shape[-1])

            if self.pos_1LC is not None:
                x_BLC += (
                    self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))
                    + self.pos_1LC[:, :ed]
                )  # lvl: BLC;  pos: 1LC
            else:
                x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x_BLC = self._gradient_checkpointing_func(
                    b.forward_function,
                    x_BLC,
                    cond_BD_or_gss,
                    attn_bias,
                    mask_for_attn,
                    context_position_ids,
                    context_mask,
                )
            else:
                x_BLC = b(
                    x=x_BLC,
                    cond_BD=cond_BD_or_gss,
                    attn_bias=attn_bias,
                    m_maskgit=mask_for_attn,
                    context_position_ids=context_position_ids,
                    context_mask=context_mask,
                )
        # parallel generation of discrete and continuous tokens
        x_BLC_logits, last_layer_cond = (
            x_BLC,
            x_BLC[:, self.L + self.context_token - 1 - self.last_level_pns :, :],
        )

        x_BLC_logits = self.get_logits(x_BLC_logits.float(), cond_BD)
        with torch.no_grad():
            # important to clone the last stage logits
            # Haotian: autoregressive LLM sometimes have this error:
            # RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
            try:
                idx_BL_sampled = sample_with_top_k_top_p_(
                    x_BLC_logits[
                        :, self.L + self.context_token - 1 - self.last_level_pns :
                    ]
                    .clone()
                    .detach(),
                    rng=self.rng,
                    top_k=600,
                    top_p=0.96,
                    num_samples=1,
                )[:, :, 0]
            except:
                idx_BL_sampled = last_layer_gt_discrete
        last_stage_discrete_embed = self.vae_quant_proxy[0].embedding(idx_BL_sampled)
        last_stage_discrete_cond = self.word_embed(last_stage_discrete_embed)
        last_layer_cond = self.decoder_norm(last_layer_cond + last_stage_discrete_cond)

        last_layer_gt_continuous = last_layer_gt - last_stage_discrete_embed
        diff_loss = self.forward_diff_loss(
            z=last_layer_cond, target=last_layer_gt_continuous
        )
        # Haotian: important, we should start from self.context_token - 1.
        return (
            x_BLC_logits[:, self.context_token - 1 :, :],
            diff_loss,
            mask_wo_prev_stages,
        )  # logits BLV, V is vocab_size

    def init_weights(
        self,
        init_adaln=0.5,
        init_adaln_gamma=1e-5,
        init_head=0.02,
        init_std=0.02,
        conv_std_or_gain=0.02,
    ):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        print(f"[init_weights] {type(self).__name__} with {init_std=:g}")
        for m in self.modules():
            with_weight = hasattr(m, "weight") and m.weight is not None
            with_bias = hasattr(m, "bias") and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(
                m,
                (
                    nn.LayerNorm,
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.SyncBatchNorm,
                    nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    m.weight.data.fill_(1.0)
                if with_bias:
                    m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(
                m,
                (
                    nn.Conv1d,
                    nn.Conv2d,
                    nn.Conv3d,
                    nn.ConvTranspose1d,
                    nn.ConvTranspose2d,
                    nn.ConvTranspose3d,
                ),
            ):
                if conv_std_or_gain > 0:
                    nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else:
                    nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias:
                    m.bias.data.zero_()

        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()

        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if (
                hasattr(self.head_nm.ada_lin[-1], "bias")
                and self.head_nm.ada_lin[-1].bias is not None
            ):
                self.head_nm.ada_lin[-1].bias.data.zero_()

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, "fc2"):
                sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, "fcg") and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, "ada_lin"):
                sab.ada_lin[-1].weight.data[2 * self.C :].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[: 2 * self.C].mul_(init_adaln_gamma)
                if (
                    hasattr(sab.ada_lin[-1], "bias")
                    and sab.ada_lin[-1].bias is not None
                ):
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, "ada_gss"):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)

        self.diffloss.initialize_weights()

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"


AutoConfig.register("hart_transformer_t2i", HARTForT2IConfig)
AutoModel.register(HARTForT2IConfig, HARTForT2I)
