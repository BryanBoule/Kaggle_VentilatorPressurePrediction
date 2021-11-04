import numpy as np
import torch
from torch.utils.data import DataLoader


def predict(
        model,
        dataset,
        batch_size=64,
        device="cuda"
):
    """
    Args:
        model (torch model): Model to predict with.
        dataset (PathologyDataset): Dataset to predict on.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(dataset) x num_classes] preds.
    """

    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    preds = []
    with torch.no_grad():
        for data in loader:
            pred = model(data['input'].to(device)).squeeze(-1)
            preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    return preds
