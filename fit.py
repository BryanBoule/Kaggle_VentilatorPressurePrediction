import gc

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from methods import VentilatorLoss, compute_metric
import time
import numpy as np


def fit(
        model,
        train_dataset,
        val_dataset,
        loss_name="L1Loss",
        optimizer="Adam",
        epochs=56,
        batch_size=32,
        val_bs=32,
        warmup_prop=0.1,
        lr=1e-3,
        num_classes=1,
        verbose=1,
        first_epoch_eval=0,
        device="cuda"
):
    avg_val_loss = 0.

    # Optimizer
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=lr)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        # how many samples per batch to load
        batch_size=batch_size,
        # to have the data reshuffled at every epoch
        shuffle=True,
        # drop the last incomplete batch
        drop_last=True,
        num_workers=4,
        # the data loader will copy Tensors into CUDA pinned memory before returning them.
        pin_memory=True,
        # this will be called on each worker subprocess with the worker id (an int in
        # ``[0, num_workers - 1]``) as input, after seeding and before data loading.
        # worker_init_fn=worker_init_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        # no shuffling val data at every epochs meaning validating on same batches of data
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Loss
    loss_fct = VentilatorLoss()

    # Scheduler
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader))
    num_training_steps = int(epochs * len(train_loader))
    # Create a schedule with a varying learning rate
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps,
        num_training_steps
    )

    for epoch in range(epochs):
        model.train()
        # Sets the gradients of all optimized Tensors to zero.
        model.zero_grad()
        start_time = time.time()

        avg_loss = 0

        for data in train_loader:

            pred = model(data['input'].to(device)).squeeze(-1)

            loss = loss_fct(
                pred,
                data['p'].to(device),
                data['u_out'].to(device),
            ).mean()
            # Computes the gradient of current tensor w.r.t.
            # (with respect to) graph leaves.
            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            scheduler.step()

            for param in model.parameters():
                param.grad = None

        model.eval()
        mae, avg_val_loss = 0, 0
        preds = []

        # Context-manager that disable gradient calculation.
        # To use when dealing with validation datasets
        with torch.no_grad():
            for data in val_loader:
                pred = model(data['input'].to(device)).squeeze(-1)

                loss = loss_fct(
                    pred.detach(),
                    data['p'].to(device),
                    data['u_out'].to(device),
                ).mean()
                avg_val_loss += loss.item() / len(val_loader)

                preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds, 0)
        mae = compute_metric(val_dataset.df_grouped, preds)

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} "
                f"\t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )

            if (epoch + 1 >= first_epoch_eval) or (epoch + 1 == epochs):
                print(f"val_loss={avg_val_loss:.3f}\tmae={mae:.3f}")
            else:
                print("")

    del (val_loader, train_loader, loss, data, pred)
    gc.collect()
    torch.cuda.empty_cache()

    return preds
