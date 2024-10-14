from typing import Optional, Tuple

from transformers import PretrainedConfig

__all__ = ["HARTAutoEncoderConfig", "HARTAutoEncoderWithDiscConfig"]


class HARTAutoEncoderConfig(PretrainedConfig):
    model_type = "hart_autoencoder"

    def __init__(
        self,
        vocab_size=4096,
        z_channels=32,
        ch=160,
        dropout=0.0,
        beta=0.25,
        using_znorm=False,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=4,
        default_qresi_counts=0,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        test_mode=False,
        ch_mult=(1, 1, 2, 2, 4),
        levels=[8, 8, 8, 6, 5],
        quantizer_type: str = "var",
        hybrid: bool = False,
        disable_quant_resi: bool = False,
        freeze_codebook_for_hybrid: bool = True,
        double_decoder=False,
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.z_channels = z_channels
        self.ch = ch
        self.dropout = dropout
        self.beta = beta
        self.using_znorm = using_znorm
        self.quant_conv_ks = quant_conv_ks
        self.quant_resi = quant_resi
        self.share_quant_resi = share_quant_resi
        self.default_qresi_counts = default_qresi_counts
        self.v_patch_nums = v_patch_nums
        self.test_mode = test_mode
        self.ch_mult = ch_mult
        self.levels = levels
        self.quantizer_type = quantizer_type
        self.hybrid = hybrid
        self.disable_quant_resi = disable_quant_resi
        self.freeze_codebook_for_hybrid = freeze_codebook_for_hybrid
        self.double_decoder = double_decoder


class HARTAutoEncoderWithDiscConfig(HARTAutoEncoderConfig):
    model_type = "hart_autoencoder_with_disc"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
