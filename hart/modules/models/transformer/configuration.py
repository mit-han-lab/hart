from typing import Optional

from transformers import PretrainedConfig

__all__ = [
    "VARTransformerConfig",
    "VARTransformerT2IConfig",
    "HARTForC2IConfig",
    "HARTForT2IConfig",
]


class VARTransformerConfig(PretrainedConfig):
    model_type = "var_transformer"

    def __init__(
        self,
        vae_path: Optional[str] = None,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True,
        fused_if_available=True,
        mlp_type="gpt2",
        attn_type="gpt2",
        disable_aln=False,
        use_timestep_embed=False,
        sep_aln_pooling_mode="max",
        use_cross_attn=False,
        latent_condition_weight=1.0,
        **kwargs,
    ):
        super().__init__()

        self.vae_path = vae_path
        self.num_classes = num_classes
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_eps = norm_eps
        self.shared_aln = shared_aln
        self.cond_drop_rate = cond_drop_rate
        self.attn_l2_norm = attn_l2_norm
        self.patch_nums = patch_nums
        self.flash_if_available = flash_if_available
        self.fused_if_available = fused_if_available
        self.mlp_type = mlp_type
        self.attn_type = attn_type
        self.disable_aln = disable_aln
        self.use_timestep_embed = use_timestep_embed
        self.sep_aln_pooling_mode = sep_aln_pooling_mode
        self.use_cross_attn = use_cross_attn
        self.diffusion_head_repeats = kwargs.get("diffusion_head_repeats", 1)
        self.latent_condition_weight = latent_condition_weight


class VARTransformerT2IConfig(PretrainedConfig):
    model_type = "var_transformer_t2i"

    def __init__(
        self,
        vae_path: Optional[str] = None,
        context_token=77,
        context_dim=768,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True,
        fused_if_available=True,
        mlp_type="gpt2",
        attn_type="gpt2",
        disable_aln=False,
        use_timestep_embed=False,
        sep_aln_pooling_mode="max",
        use_cross_attn=False,
        **kwargs,
    ):
        super().__init__()

        self.vae_path = vae_path
        self.context_token = context_token
        self.context_dim = context_dim
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_eps = norm_eps
        self.shared_aln = shared_aln
        self.cond_drop_rate = cond_drop_rate
        self.attn_l2_norm = attn_l2_norm
        self.patch_nums = patch_nums
        self.flash_if_available = flash_if_available
        self.fused_if_available = fused_if_available
        self.mlp_type = mlp_type
        self.attn_type = attn_type
        self.disable_aln = disable_aln
        self.use_timestep_embed = use_timestep_embed
        self.sep_aln_pooling_mode = sep_aln_pooling_mode
        self.use_cross_attn = use_cross_attn


class HARTForC2IConfig(VARTransformerConfig):
    model_type = "hart_transformer_c2i"

    def __init__(
        self,
        diff_width=1024,
        diff_depth=6,
        num_sampling_steps="8",
        diffusion_batch_mul=4,
        sampler="iddpm",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diff_width = diff_width
        self.diff_depth = diff_depth
        self.num_sampling_steps = num_sampling_steps
        self.diffusion_batch_mul = diffusion_batch_mul
        self.sampler = sampler


class HARTForT2IConfig(VARTransformerT2IConfig):
    model_type = "hart_transformer_t2i"

    def __init__(
        self,
        diff_width=1024,
        diff_depth=6,
        num_sampling_steps="8",
        diffusion_batch_mul=4,
        sampler="iddpm",
        use_context_norm=False,
        context_norm_scale=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.diff_width = diff_width
        self.diff_depth = diff_depth
        self.num_sampling_steps = num_sampling_steps
        self.diffusion_batch_mul = diffusion_batch_mul
        self.sampler = sampler
        self.diffusion_head_repeats = kwargs.get("diffusion_head_repeats", 1)
        self.use_context_norm = use_context_norm
        self.context_norm_scale = context_norm_scale
