import torch


class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # k-fold
    k = 7
    selected_folds = [0]

    # Model
    selected_model = 'LSTM'
    input_dim = 5  # 30

    dense_dim = 512
    lstm_dim = 512
    logit_dim = 512
    num_classes = 1

    # Training
    loss = "L1Loss"  # not used
    optimizer = "Adam"
    batch_size = 256
    epochs = 100

    lr = 1e-3
    warmup_prop = 0

    val_bs = 256
    first_epoch_eval = 0