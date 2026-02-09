from typing import Optional

from transformers import PretrainedConfig


class NanovlmConfig(PretrainedConfig):
    model_type = "nanovlm"

    def __init__(
        self,
        vision_model_name: str = "google/siglip2-base-patch16-naflex",
        llm_model_name: str = "Qwen/Qwen3-0.6B",
        image_token_id: Optional[int] = None,
        projector_hidden_size: Optional[int] = None,
        projector_num_layers: int = 2,
        projector_hidden_act: str = "gelu",
        vision_feature_dim: int = 1152,
        image_token_count: int = 256,
        **kwargs,
    ):
        self.vision_model_name = vision_model_name
        self.llm_model_name = llm_model_name
        self.image_token_id = image_token_id
        self.projector_hidden_size = projector_hidden_size
        self.projector_num_layers = int(projector_num_layers)
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_dim = int(vision_feature_dim)
        self.image_token_count = int(image_token_count)
        super().__init__(**kwargs)
