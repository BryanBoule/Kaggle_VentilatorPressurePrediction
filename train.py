import gc

import torch

from biGRU_3 import GRU_archi
from biLSTM_3 import LSTM_archi_2
from simple_biLSTM import LSTM_archi
from dataloader import VPPDataLoader
from methods import seed_everything, save_model_weights
from fit import fit
from predict import predict


def train(config, df_train, df_val, df_test, fold):
    """
    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        df_test (pandas dataframe): Test metadata.
        fold (int): Selected fold.

    Returns:
        np array: Study validation predictions.
    """

    # Seed
    seed_everything(config.seed)

    # Load model arch
    model = LSTM_archi_2(
        input_dim=config.input_dim,
        lstm_dim=config.lstm_dim,
        dense_dim=config.dense_dim,
        logit_dim=config.logit_dim,
        num_classes=config.num_classes,
    ).to(config.device)
    model.zero_grad()

    train_dataset = VPPDataLoader(df_train)
    val_dataset = VPPDataLoader(df_val)
    test_dataset = VPPDataLoader(df_test)

    print(f"    -> {len(train_dataset)} training breathes")
    print(f"    -> {len(val_dataset)} validation breathes")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        loss_name=config.loss,
        optimizer=config.optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        warmup_prop=config.warmup_prop,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
    )

    pred_test = predict(
        model,
        test_dataset,
        batch_size=config.val_bs,
        device=config.device
    )

    if config.save_weights:
        save_model_weights(
            model,
            f"{config.selected_model}_{fold}.pt",
            cp_folder="./models",
        )

    del (model, train_dataset, val_dataset, test_dataset)
    gc.collect()
    torch.cuda.empty_cache()

    return pred_val, pred_test
