import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback


class IntervalModelCheckpoint(Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
            self,
            dirpath,
            save_intervals,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.dirpath = dirpath
        self.save_intervals = save_intervals
        self.best_val_loss = 1e10

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        global_step = trainer.global_step

        if (global_step + 1) in self.save_intervals:
            trainer.run_evaluation()
            val_loss = trainer.callback_metrics['val_loss']
            filename = f"steps={global_step+1:05d}-val_loss={val_loss:0.8f}.ckpt"
            ckpt_path = os.path.join(self.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

            if val_loss < self.best_val_loss:
                best_ckpt_path = os.path.join(self.dirpath, 'best.ckpt')
                trainer.save_checkpoint(best_ckpt_path)
                self.best_val_loss = val_loss
