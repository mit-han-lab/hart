from transformers import PretrainedConfig

__all__ = ["NLayerDiscriminatorConfig"]


class NLayerDiscriminatorConfig(PretrainedConfig):
    model_type = "NLayerDiscriminator"

    def __init__(
        self,
        input_nc=3,
        ndf=64,
        n_layers=3,
        use_actnorm=False,
        **kwargs,
    ):
        super().__init__()

        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers
        self.use_actnorm = use_actnorm
