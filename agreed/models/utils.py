from typing import Tuple, List, Union

import torch
from tqdm import tqdm


def evaluate_model(model: torch.nn.Module, 
                   data_loaders: Union[List[torch.utils.data.DataLoader], torch.utils.data.DataLoader], 
                   device: torch.device,
                   return_predictions: bool = False):
    """
    Evaluate the model on train/valid/test data.
    """

    if isinstance(data_loaders, list):
        errors = torch.zeros(len(data_loaders))
    else:
        data_loaders = [data_loaders]
        errors = torch.zeros(1)

    with torch.no_grad():
        pred_list=[]
        for i, data_loader in enumerate(data_loaders):
            predictions=[]
            n_examples = len(data_loader.dataset)
            for (x, y) in data_loader:
                y_pred = model(x.to(device))
                y_pred = y_pred.cpu()
                if return_predictions:
                    predictions.append(y_pred)
                y_pred = torch.argmax(y_pred, axis=-1)
                errors[i] += (y_pred != y).float().sum()
            if return_predictions:
                pred_list.append(torch.cat(predictions, 0))
            errors[i] /= n_examples

    # Return an error rate
    errors *= 100

    # Return predictions if requested
    if return_predictions:
        return errors, pred_list
    return errors


def eval_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
               device: torch.device) -> Tuple[torch.Tensor, float]:
    model.to(device)
    model.eval()

    accuracies = []
    correctly_classified = 0
    totally_classified = 0
    with torch.no_grad():
        for inputs, tgts in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs).to(torch.device("cpu"))
            accuracies.append(torch.softmax(outputs, dim=1)[torch.arange(0, tgts.shape[0], 1, dtype=torch.int64), tgts.to(torch.int64)])
            correctly_classified += (torch.argmax(outputs, dim=1) == tgts).sum().item()
            totally_classified += tgts.numel()

    model.to(torch.device("cpu"))
    inputs = inputs.to(torch.device("cpu"))
    torch.cuda.empty_cache()

    return torch.cat(accuracies), correctly_classified / totally_classified


def eval_model_scores(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    model.to(device)
    model.eval()

    model_scores = []
    with torch.no_grad():
        for batch_inputs, _ in dataloader:
            batch_inputs = batch_inputs.to(device)

            outputs = model(batch_inputs).to(torch.device("cpu"))
            model_scores.append(outputs)

    model.to(torch.device("cpu"))
    batch_inputs = batch_inputs.to(torch.device("cpu"))
    torch.cuda.empty_cache()

    model_scores = torch.cat(model_scores)

    return model_scores


def eval_model_probes(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    model.to(device)
    model.eval()

    model_probes = []
    with torch.no_grad():
        for batch_inputs, _ in dataloader:
            batch_inputs = batch_inputs.to(device)

            outputs = model(batch_inputs).to(torch.device("cpu"))
            model_probes.append(torch.softmax(outputs, dim=1))

    model.to(torch.device("cpu"))
    batch_inputs = batch_inputs.to(torch.device("cpu"))
    torch.cuda.empty_cache()

    model_probes = torch.cat(model_probes)

    return model_probes
