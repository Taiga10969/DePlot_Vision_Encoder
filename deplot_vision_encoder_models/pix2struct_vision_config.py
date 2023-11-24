import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)

class Pix2StructVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Pix2StructVisionModel`]. It is used to
    instantiate a Pix2Struct vision model according to the specified arguments, defining the model architecture.
    Instantiating a configuration defaults will yield a similar configuration to that of the Pix2Struct-base
    [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        patch_embed_hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the input patch_embedding layer in the Transformer encoder.
        d_ff (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        d_kv (`int`, *optional*, defaults to 64):
            Dimensionality of the key, query, value projections per attention head.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        dense_act_fn (`str` or `function`, *optional*, defaults to `"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        dropout_rate (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 1e-10):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        seq_len (`int`, *optional*, defaults to 4096):
            Maximum sequence length (here number of patches) supported by the model.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance (in tokens) to use for each attention layer.

    Example:

    ```python
    >>> from transformers import Pix2StructVisionConfig, Pix2StructVisionModel

    >>> # Initializing a Pix2StructVisionConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructVisionConfig()

    >>> # Initializing a Pix2StructVisionModel (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "pix2struct_vision_model"

    def __init__(
        self,
        hidden_size=768,
        patch_embed_hidden_size=768,
        d_ff=2048,
        d_kv=64,
        num_hidden_layers=12,
        num_attention_heads=12,
        dense_act_fn="gelu_new",
        layer_norm_eps=1e-6,
        dropout_rate=0.0,
        attention_dropout=0.0,
        initializer_range=1e-10,
        initializer_factor=1.0,
        seq_len=4096,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.patch_embed_hidden_size = patch_embed_hidden_size
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.dense_act_fn = dense_act_fn
        self.seq_len = seq_len
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.d_kv = d_kv

    @classmethod
    def from_pretrained(
        cls, pretrainehidden_size_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> "PretrainedConfig":
        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrainehidden_size_name_or_path, **kwargs)

        # get the vision config dict if we are loading from Pix2StructConfig
        if config_dict.get("model_type") == "pix2struct":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

if __name__ == '__main__':

    vision_config = Pix2StructVisionConfig.from_pretrained('google/deplot')
    
    print('Pix2StructForConditionalGeneration >> type(vision_config) : ', type(vision_config))
    print('Pix2StructForConditionalGeneration >> vision_config : ', vision_config)
