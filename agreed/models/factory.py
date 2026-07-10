import torch
import torch.nn as nn
from torchvision.models.convnext import convnext_tiny
from torchvision.models.mobilenetv2 import mobilenet_v2

from agreed.models.resnet import resnet18, resnet50
from agreed.models.vgg import VGG
from agreed.dataset.images.utils import MINI_IMAGENET_CLS_TO_IMAGENET_CLS
from agreed.config.utils import DataConfig, ModelConfig


class ModelFactory:
    @staticmethod
    def init_model(
        model_config: ModelConfig, dataset_config: DataConfig, load_weights: bool = True
    ) -> nn.Module:

        model: nn.Module

        if model_config.name.lower() == "resnet18":
            model = resnet18(num_classes=dataset_config.num_classes)
        elif model_config.name.lower() == "resnet50":
            model = resnet50(num_classes=dataset_config.num_classes)
        elif model_config.name.lower() == "mobilenetv2":
            model = mobilenet_v2(num_classes=dataset_config.num_classes)
        elif model_config.name.lower() == "vgg16":
            if "imagenet" in dataset_config.name.lower():
                num_classes = 1000
            else:
                num_classes = dataset_config.num_classes
            model = VGG(
                num_classes=num_classes,
                model_name=model_config.name,
                dropout=model_config.dropout,
                init_weights=True,
                batch_norm=model_config.batch_norm,
                adaptive_pool_size=model_config.adaptive_pool_size,
                layerwidths=model_config.layerwidths,
            )
        elif model_config.name.lower() == "convnext":
            if "imagenet" in dataset_config.name.lower():
                model = convnext_tiny(num_classes=1000)
            else:
                raise Exception("Convnext only available on Imagenet")
        else:
            raise NotImplementedError(
                f"'{model_config.name}' model is not implemented for this project."
            )

        if load_weights:
            model.load_state_dict(
                torch.load(model_config.checkpoints, weights_only=True)
            )

        # TODO For miniimagenet, we might need to prune the last layer to have 100 classes instead of 1K
        if dataset_config.name.lower() == "miniimagenet":
            split_last_layer_name = model_config.last_layer_name.split(".")
            last_layer = model
            for module_name in split_last_layer_name:
                last_layer = last_layer._modules[module_name]
            if isinstance(last_layer, nn.Linear):
                last_layer.weight = torch.nn.Parameter(
                    last_layer.weight[MINI_IMAGENET_CLS_TO_IMAGENET_CLS]
                )
                last_layer.bias = torch.nn.Parameter(
                    last_layer.bias[MINI_IMAGENET_CLS_TO_IMAGENET_CLS]
                )
                last_layer.out_features = 100

        return model
