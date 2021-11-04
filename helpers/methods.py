import numpy as np
import torch.nn as nn
import random
import os
import torch


def create_features(dataframe, list_of_features=['u_in']):
    # u_in cumsum
    dataframe['u_in_cumsum'] = dataframe.groupby('breath_id')['u_in'].cumsum()

    # u_in shift change
    for lag in np.arange(1, 5, 1):
        dataframe[f'u_in_lag_fwrd{lag}'] = dataframe.groupby('breath_id')['u_in'].shift(
            lag).fillna(0)
        dataframe[f'u_in_lag_back{lag}'] = dataframe.groupby('breath_id')['u_in'].shift(
            int(-lag)).fillna(0)

    # time diff
    dataframe['time_diff'] = dataframe.groupby('breath_id')['time_step'].diff(1).fillna(0)
    dataframe['time_diff_2'] = dataframe.groupby('breath_id')['time_step'].diff(2).fillna(
        0)
    dataframe['time_diff_3'] = dataframe.groupby('breath_id')['time_step'].diff(3).fillna(
        0)
    dataframe['time_diff_4'] = dataframe.groupby('breath_id')['time_step'].diff(4).fillna(
        0)
    dataframe['time_diff_5'] = dataframe.groupby('breath_id')['time_step'].diff(5).fillna(
        0)

    # u_in area
    dataframe['area'] = dataframe['time_step'] * dataframe['u_in']
    dataframe['area_cumsum'] = dataframe.groupby('breath_id')['area'].cumsum()
    # add rectangle method
    dataframe['auc_u_in'] = dataframe['time_diff'] * dataframe['u_in']
    dataframe['auc_u_in_cumsum'] = dataframe.groupby('breath_id')['auc_u_in'].cumsum()

    dataframe['u_in_cumsum'] = dataframe.groupby('breath_id')['u_in'].cumsum()

    for feature in list_of_features:
        grouped_dataframe = dataframe.groupby('breath_id')[feature].agg(
            [max, min, np.mean, np.median])

        dataframe = dataframe.merge(
            grouped_dataframe,
            how='left',
            on='breath_id'
        )

        dataframe = dataframe.rename(
            columns={
                'max': feature + '_max',
                'min': feature + '_min',
                'mean': feature + '_mean',
                'median': feature + '_median'
            }
        )

        dataframe[f'{feature}_range'] = (
                    dataframe[f'{feature}_max'] - dataframe[f'{feature}_min']).apply(
            lambda x: max(0, x))

    return dataframe


# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models

# next time, fool, use directly the nn.SiLU() activation

class Swish(nn.Module):
    '''
    Applies the Sigmoid Linear Unit (SiLU) function element-wise:
        SiLU(x) = x * sigmoid(x)
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    '''

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results.

    Args:
        seed (int): Number of the seed.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    """
    Handles PyTorch x Numpy seeding issues.

    Args:
        worker_id (int): Id of the worker.
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def save_model_weights(model, filename, verbose=1, cp_folder=""):
    """
    Saves the weights of a PyTorch model.

    Args:
        model (torch model): Model to save the weights of.
        filename (str): Name of the checkpoint.
        verbose (int, optional): Whether to display infos. Defaults to 1.
        cp_folder (str, optional): Folder to save to. Defaults to "".
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def compute_metric(df, preds):
    """
    Metric for the problem, as I understood it.
    """

    y = np.array(df['pressure'].values.tolist())

    # inspiratory phase
    mask = 1 - np.array(df['u_out'].values.tolist())

    # combine with mae calculusse
    mae = mask * np.abs(y - preds)
    mae = mae.sum() / mask.sum()

    return mae


# Custom loss
class VentilatorLoss(nn.Module):
    """
    Directly optimizes the competition metric
    """

    def __call__(self, preds, y, u_out):
        mask = 1 - u_out
        mae = mask * (y - preds).abs()
        mae = mae.sum(-1) / mask.sum(-1)

        return mae
