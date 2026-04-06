import os
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import mark_boundaries
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from agreed.config.utils import ImageDataConfig, ModelConfig
from agreed.dataset.images.factory import ImageDatasetFactory
from agreed.models.factory import ModelFactory

PATH_CONFIG = os.path.join("..", "..", "agreed", "config")


def main():

    from simple_parsing import ArgumentParser

    ################################### Setup #################################

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="The device on which to run inference")
    parser.add_argument("--model", type=str, default="ResNet18", help="The model to explain : ResNet18, Resnet50, MobileNetv2, ConvNext")
    args, unknown = parser.parse_known_args()
    print(args)

    # Parse the Dataset configuration file and load the data
    with open(os.path.join(PATH_CONFIG, "datasets", "MiniImageNet.yaml"), "r") as f:
        data_config = ImageDataConfig(**yaml.load(f, Loader=yaml.FullLoader))
    _, val_dataset = ImageDatasetFactory().load_dataset(data_config)

    # Parse the Model configuration file and load the model
    with open(os.path.join(PATH_CONFIG, "models", f"{args.model}.yaml"), "r") as f:
        model_config = ModelConfig(**yaml.load(f, Loader=yaml.FullLoader))
        # If it is a f-string
        if model_config.checkpoints[0] == "f":
            model_config.checkpoints = eval(model_config.checkpoints)
    model = ModelFactory().init_model(model_config, data_config).eval()
    model = model.to(args.device)

    test_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False, num_workers=4)

    count = 0
    for x, targets in tqdm(test_loader):
        with torch.no_grad():
            out = model(x.to(args.device)).cpu()
            out = out.argmax(dim=1)
            correct_pred_idx = torch.where(out == targets)[0]
            count += len(correct_pred_idx)
    print(count / len(val_dataset))

if __name__ == "__main__":
    main()
