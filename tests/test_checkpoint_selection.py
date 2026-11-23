import logging
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import torch

from synapse.main import train


class EpochModel(L.LightningModule):
    """One SGD update per epoch makes the tested checkpoint observable in its prediction."""

    def __init__(self, metrics, **kwargs):
        super().__init__()
        self.metrics = metrics
        self.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.model.weight)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1.0)

    def training_step(self, batch, batch_idx):
        return self.model(batch).mean()

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", float(abs(self.current_epoch - 1)))
        self.log("val_quality_epoch", float(2 - self.current_epoch))

    def test_step(self, batch, batch_idx):
        self.trainer.datamodule.test_prediction = self.model(batch).item()


class CheckpointSelectionTests(unittest.TestCase):
    def test_post_fit_scoring_uses_monitored_epoch(self):
        quality = {"stages": ["val"], "on_epoch": True, "is_monitor": True, "mode": "max"}
        # Predictions at epochs 0, 1 and 2 are -1, -2 and -3 respectively.
        for metrics, expected in [({}, -2.0), ({"quality": quality}, -1.0)]:
            with self.subTest(metrics=metrics), tempfile.TemporaryDirectory() as run_dir:
                sample = torch.ones(1, 1)
                data = L.LightningDataModule.from_datasets(
                    train_dataset=sample, val_dataset=sample, test_dataset=sample, batch_size=1,
                )
                run = SimpleNamespace(
                    run_mode=["train", "test"], run_dir=run_dir, checkpoint_dir="checkpoints",
                    seed=42, device="cpu", n_devices=1, epochs=3, use_amp=False,
                    val_sanity_check=False, test_output=None, onnx_path=None,
                )
                model = SimpleNamespace(
                    load_model=None, model="EpochModel", model_params={}, metrics=metrics,
                    optimizer="SGD", start_lr=1.0, lr_scheduler=None,
                    loss_function={"name": "mean", "params": {}},
                )
                with patch("synapse.main.ModelModule", EpochModel), patch("synapse.main.DataModule", return_value=data):
                    train(None, model, None, run, [], [], [], logging.getLogger(__name__), "test")
                self.assertEqual(data.test_prediction, expected)
