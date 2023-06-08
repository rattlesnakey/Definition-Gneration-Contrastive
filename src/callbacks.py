import logging
import os
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)

def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    if metric == "loss":
        exp = "{val_avg_loss:.4f}-{epoch_count}"

    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        period=0,  # maybe save a checkpoint every time val is run, not just end of epoch.
    )
    return checkpoint_callback


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )
