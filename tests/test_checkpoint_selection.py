import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import torch
import yaml

from synapse.core.config import RunConfig
from synapse.core.model_module import ModelModule
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


class AdamEpochModel(EpochModel):
    on_train_epoch_start = ModelModule.on_train_epoch_start

    def __init__(self, run_cfg, **kwargs):
        super().__init__(**kwargs)
        self.run_cfg = run_cfg

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=.01)
        return [opt], [torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.run_cfg.epochs)]


class CheckpointSelectionTests(unittest.TestCase):
    def test_extension_restores_adam_steps_and_starts_new_cosine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = torch.ones(1, 1)
            model = SimpleNamespace(load_model=None, model="AdamEpochModel", model_params={}, metrics={},
                                    optimizer="adam", start_lr=.01, lr_scheduler="cosine",
                                    loss_function={"name": "mean", "params": {}})
            for total in (2, 4):
                directory = root / str(total)
                params = dict(run_dir=str(directory), run_mode=["train"], epochs=total, device="cpu",
                              n_devices=1, batch_size=1, seed=42, checkpoint_dir="checkpoints", logger_config={})
                if total == 4:
                    last, = (root / "2/checkpoints").glob("model-epoch=epoch=1-*.ckpt")
                    params.update(resume_checkpoint=str(last), cosine_restart={"epoch": 2, "lr": .001})
                cfg = root / "run.yaml"
                cfg.write_text(yaml.safe_dump(params))
                data = L.LightningDataModule.from_datasets(train_dataset=sample, val_dataset=sample, batch_size=1)
                with patch("synapse.main.ModelModule", AdamEpochModel), patch("synapse.main.DataModule", return_value=data):
                    train(None, model, None, RunConfig(str(cfg)), [], [], [], logging.getLogger(__name__), "test")
            original = torch.load(last, map_location="cpu", weights_only=False)
            self.assertEqual(original["optimizer_states"][0]["param_groups"][0]["lr"], 0.)
            for epoch, lr in [(2, .0005), (3, 0.)]:
                path, = (root / "4/checkpoints").glob(f"model-epoch=epoch={epoch}-*.ckpt")
                resumed = torch.load(path, map_location="cpu", weights_only=False)
                opt = resumed["optimizer_states"][0]
                self.assertEqual(resumed["global_step"], epoch + 1)
                self.assertEqual(opt["state"][0]["step"].item(), epoch + 1)
                self.assertAlmostEqual(opt["param_groups"][0]["lr"], lr)
            torch.testing.assert_close(resumed["state_dict"]["model.weight"],
                                       original["state_dict"]["model.weight"] - .0015)

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
