import os
from omegaconf import OmegaConf, MISSING
from dataclasses import dataclass, field


PATH_CONFIG = os.path.dirname(os.path.abspath(__file__))

@dataclass
class ModelConfig:
    name: str = MISSING
    checkpoints: str = MISSING
    last_layer_name: str = MISSING

@dataclass
class DataConfig:
    name: str = MISSING
    imsize: tuple[int, int] = MISSING
    num_classes: int = MISSING
    in_channels: int = MISSING
    path: str = MISSING

@dataclass
class Config:
    save_path: str
    data_path: str
    model_path: str
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DataConfig = field(default_factory=DataConfig)


def get_config(dataset: str, model: str) -> Config:

    schema = OmegaConf.structured(Config)
    base_config = OmegaConf.load(os.path.join(PATH_CONFIG, "base.yaml"))
    data_config = OmegaConf.load(os.path.join(PATH_CONFIG, "datasets", f"{dataset}.yaml"))
    model_config = OmegaConf.load(os.path.join(PATH_CONFIG, "models", f"{model}.yaml"))
    base_config.dataset = data_config
    base_config.model = model_config
    new_config: Config = OmegaConf.to_object(OmegaConf.merge(schema, base_config))

    # Some runtime conversions are needed
    new_config.dataset.imsize = tuple(new_config.dataset.imsize)

    return new_config
