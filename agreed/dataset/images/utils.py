import os
from glob import glob
from typing import Any, List, Sequence, Tuple
from pathlib import Path
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler, random_split, TensorDataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


MINI_IMAGENET_CLS_TO_IMAGENET_CLS = [12, 15, 51, 64, 70, 96, 99, 107, 111, 121, 149, 166, 173, 176, 207, 214, 228, 242,
                                     244, 245, 249, 251, 256, 266, 270, 275, 279, 291, 299, 301, 306, 310, 359, 364,
                                     392, 403, 412, 427, 440, 454, 471, 476, 478, 484, 494, 502, 503, 507, 519, 524,
                                     533, 538, 546, 553, 556, 567, 569, 584, 597, 602, 604, 605, 629, 655, 657, 659,
                                     683, 687, 702, 709, 713, 735, 741, 758, 779, 781, 800, 801, 807, 815, 819, 847,
                                     854, 858, 860, 880, 881, 883, 909, 912, 914, 919, 925, 927, 934, 950, 972, 973,
                                     997, 998]

MINI_IMAGENET_CLS_TO_IMAGENET_NAME = [
    "house_finch",
    "robin",
    "triceratops",
    "green_mamba",
    "harvestman",
    "toucan",
    "goose",
    "jellyfish",
    "nematode",
    "king_crab",
    "dugong",
    "Walker_hound",
    "Ibizan_hound",
    "Saluki",
    "golden_retriever",
    "Gordon_setter",
    "komondor",
    "boxer",
    "Tibetan_mastiff",
    "French_bulldog",
    "malamute",
    "dalmatian",
    "Newfoundland",
    "miniature_poodle",
    "white_wolf",
    "African_hunting_dog",
    "Arctic_fox",
    "lion",
    "meerkat",
    "ladybug",
    "rhinoceros_beetle",
    "ant",
    "black-footed_ferret",
    "three-toed_sloth",
    "rock_beauty",
    "aircraft_carrier",
    "ashcan",
    "barrel",
    "beer_bottle",
    "bookshop",
    "cannon",
    "carousel",
    "carton",
    "catamaran",
    "chime",
    "clog",
    "cocktail_shaker",
    "combination_lock",
    "crate",
    "cuirass",
    "dishrag",
    "dome",
    "electric_guitar",
    "file",
    "fire_screen",
    "frying_pan",
    "garbage_truck",
    "hair_slide",
    "holster",
    "horizontal_bar",
    "hourglass",
    "iPod",
    "lipstick",
    "miniskirt",
    "missile",
    "mixing_bowl",
    "oboe",
    "organ",
    "parallel_bars",
    "pencil_box",
    "photocopier",
    "poncho",
    "prayer_rug",
    "reel",
    "school_bus",
    "scoreboard",
    "slot",
    "snorkel",
    "solar_dish",
    "spider_web",
    "stage",
    "tank",
    "theater_curtain",
    "tile_roof",
    "tobacco_shop",
    "unicycle",
    "upright",
    "vase",
    "wok",
    "worm_fence",
    "yawl",
    "street_sign",
    "consomme",
    "trifle",
    "hotdog",
    "orange",
    "cliff",
    "coral_reef",
    "bolete",
    "ear",
]


def get_transforms(imsize: tuple[int, int], dataset: str = "MiniImageNet") -> transforms.Compose:
    """Create and return torchvision.transforms.Compose for the given dataset

    :param imsize: Height or Width (considered as squared image) value of the image size
    :type imsize: int
    :param dataset: Dataset name, defaults to "MiniImageNet"
    :type dataset: str, optional
    :raises NotImplementedError: Raised when given dataset is not directly handled by the function
    :return: Transformations for each image in the dataset
    :rtype: transforms.Compose
    """
    if "imagenet" in dataset.lower():
        return transforms.Compose(
            [
                transforms.Resize((imsize[0], imsize[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif "mnist" == dataset.lower():
        return transforms.Compose(
            [
                transforms.Resize((imsize[0], imsize[1])),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
    elif "cifar10" == dataset.lower():
        return transforms.Compose(
            [
                transforms.Resize((imsize[0], imsize[1])),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    else:
        raise NotImplementedError("No processing available for this specific dataset.")



def select_correctly_predicted_samples(
    model: torch.nn.Module,
    target_idx: int,
    dataset: datasets.ImageFolder,
    device: str = "cpu",
    batch_size: int = 1,
) -> torch.utils.data.Dataset:
    """Select and extract samples that are correctly predicted to be part of the chosen class

    :param model: PyTorch model
    :type model: torch.nn.Module
    :param target_idx: class indice of interest
    :type target_idx: int
    :param dataset: PyTorch Dataset module containing every samples and their target
    :type dataset: datasets.ImageFolder
    :param batch_size: Mini batch size
    :type batch_size: int
    :param device: Hardware acceleration device (cpu, gpu, tpu, hpu, etc...), defaults to "cpu"
    :type device: str, optional
    :param shuffle: DataLoader needs to be shuffled, defaults to False
    :type shuffle: bool, optional
    :param batch_size: Mini batch size for computing purpose, defaults to 1
    :type batch_size: int, optional
    :return: A tuple containing the selected samples classified easily and their targets
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """

    if hasattr(dataset, "targets"):
        target_indices = np.where(np.asarray(dataset.targets) == target_idx)[0]
    else:
        targets = np.asarray([t for _, _, t in dataset])
        target_indices = np.where(np.asarray(targets) == target_idx)[0]
        print(target_indices)

    test_loader = DataLoader(dataset=Subset(dataset, target_indices), batch_size=batch_size, shuffle=False, num_workers=4)
    samples = torch.Tensor()
    preds = torch.LongTensor()
    model = model.to(device)

    count = 0
    for x, targets in test_loader:
        with torch.no_grad():
            out = model(x.to(device)).cpu()
            out = out.argmax(dim=1)
            correct_pred_idx = torch.where(out == targets)[0]
            samples = torch.cat([samples, x[correct_pred_idx]])
            preds = torch.cat([preds, out[correct_pred_idx]])
            count += len(correct_pred_idx)
    return TensorDataset(samples, preds)


# def select_difficult_test_examples(
#     model: nn.Module,
#     target_idx: int,
#     dataset: datasets.ImageFolder,
#     batch_size: int,
#     device: str = "cpu",
#     shuffle: bool = False,
#     internal_batch_size: int = 1,
# ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
#     """Select and extract samples that are considered hard by our criteria.

#     :param model: PyTorch model
#     :type model: torch.nn.Module
#     :param target_idx: class indice of interest
#     :type target_idx: int
#     :param dataset: PyTorch Dataset module containing every samples and their target
#     :type dataset: datasets.ImageFolder
#     :param batch_size: Mini batch size
#     :type batch_size: int
#     :param device: Hardware acceleration device (cpu, gpu, tpu, hpu, etc...), defaults to "cpu"
#     :type device: str, optional
#     :param shuffle: DataLoader needs to be shuffled, defaults to False
#     :type shuffle: bool, optional
#     :param internal_batch_size: Mini batch size for computing purpose, defaults to 1
#     :type internal_batch_size: int, optional
#     :return: A tuple containing the selected samples classified with difficulty and their targets
#     :rtype: Tuple[torch.Tensor, torch.Tensor]
#     """
#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=4,
#         pin_memory=True,
#         collate_fn=None,
#     )

#     if hasattr(loader.dataset, "targets"):
#         target_indices = np.where(np.asarray(loader.dataset.targets) == target_idx)[0]
#     else:
#         targets = np.asarray([t for _, _, t in loader.dataset])
#         target_indices = np.where(np.asarray(targets) == target_idx)[0]
#         print(target_indices)

#     sampler = SubsetRandomSampler(target_indices)
#     test_loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=internal_batch_size, num_workers=4)

#     samples = torch.Tensor().to(device)
#     labels = torch.LongTensor().to(device)
#     wrong_labels_idx = []
#     probs = torch.Tensor().to(device)

#     count = 0
#     for i, (x, targets) in enumerate(test_loader):
#         with torch.no_grad():
#             x, targets = x.to(device), targets.to(device)
#             logits = model(x)
#             out = np.asarray(logits.cpu().numpy())

#             idx = np.where(np.argmax(out, axis=1) != target_idx)[0]
#             if len(idx) > 0:
#                 samples = torch.cat([samples, x[idx]])
#                 labels = torch.cat([labels, targets[idx]])
#                 probs = torch.cat([probs, F.softmax(logits[idx], dim=1)])
#                 wrong_labels_idx.append(np.argmax(out, axis=1)[idx])
#                 count += len(idx)

#     print("sample shape: ", samples.shape)
#     if len(probs) > 0:
#         probs = np.mean(probs.cpu().numpy()[:, target_idx])
#         print(f"\nModel test accuracy on difficult samples of label [{target_idx}] : {(100 * probs):.2f}%")
#     return samples, labels, wrong_labels_idx



# def filter_dataset_by_labels(dataset: Dataset, targets_idx: List[int]) -> Subset:
#     """The objective is to extract only the samples from the given labels (or targets_idx input)

#     :param dataset: PyTorch Dataset module containing every samples and their target
#     :type dataset: Dataset
#     :param targets_idx: Given labels (or classes targets)
#     :type targets_idx: List[int]
#     :return: Pytorch Subset (which inherits from Dataset module) containing only samples from given classes
#     :rtype: Subset
#     """
#     # TODO could be vectorized???
#     targets_samples_idx = [idx for idx, (_, lbl) in enumerate(dataset) if lbl in targets_idx]
#     return Subset(dataset, targets_samples_idx)


# def train_valid_split(dataset: Dataset, train_ratio: float) -> Tuple[Dataset, Dataset]:
#     """Using torch.utils.data random_split function to split the dataset into train dataset and validation dataset

#     :param dataset: PyTorch Dataset module containing every samples and their target
#     :type dataset: Dataset
#     :param train_ratio: Ratio of the portion of the dataset going to training subset
#     :type train_ratio: float
#     :return: Tuple containing training dataset and validation dataset
#     :rtype: Tuple[Dataset, Dataset]
#     """
#     total_size = len(dataset)

#     train_size = int(train_ratio * total_size)
#     valid_size = total_size - train_size

#     train_set, valid_set = random_split(dataset, [train_size, valid_size])
#     return train_set, valid_set
