import os
from pathlib import Path

from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, ImageFolder

from agreed.dataset.images.utils import get_transforms
from agreed.config.utils import DataConfig


class ImageDatasetFactory:
    """Dataset Factory used to fetch and load the correct image dataset"""

    @staticmethod
    def load_dataset(dataset_config: DataConfig, debug:bool = False) -> tuple[Dataset, Dataset]:
        """Load the correct situational dataset

        :param dataset_config: Configurations of dataset
        :type dataset_config: DatasetConfiguration
        :raises NotImplementedError: Dataset is not supported for the moment (add it if needed)
        :return: Tuple containing train dataset and validation/test dataset
        :rtype: Tuple[Dataset, Dataset]
        """
        dataset_path = Path(dataset_config.path)

        if "imagenet" in dataset_config.name.lower():
            transforms = get_transforms(dataset_config.imsize, dataset_config.name)
            train_dataset = ImageFolder(
                                root=os.path.join(dataset_config.path, "train"),
                                transform=transforms
                            )
            valid_dataset = ImageFolder(
                                root=os.path.join(dataset_config.path, "val"),
                                transform=transforms
                            )


        elif dataset_config.name.lower() == "mnist":
            transforms = get_transforms(dataset_config.imsize, dataset_config.name)

            train_dataset = MNIST(
                root=dataset_path,
                train=True,
                transform=transforms,
                download=True,
            )

            valid_dataset = MNIST(
                root=dataset_path,
                train=False,
                transform=transforms,
                download=True,
            )

        else:
            raise NotImplementedError(f"'{dataset_config.name}' dataset is not implemented for this project.")

        return train_dataset, valid_dataset
