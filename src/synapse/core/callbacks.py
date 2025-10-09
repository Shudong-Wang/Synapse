import logging
from typing import Any, Union
from typing_extensions import override

import awkward as ak
import lightning as L
import numpy as np
import torch

import lightning.pytorch as pl
from lightning.pytorch.utilities.model_summary import DeepSpeedSummary, summarize
from lightning.pytorch.utilities.model_summary import ModelSummary as Summary
from lightning.pytorch.utilities.model_summary.model_summary import _format_summary_table

from .config import DataConfig, ModelConfig, RunConfig
from .fileio import write_file

_logger = logging.getLogger("SynapseLogger")

class ModelSummary(L.Callback):
    r"""Generates a summary of all layers in a :class:`~lightning.pytorch.core.LightningModule`.

    Args:
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off.
        **summarize_kwargs: Additional arguments to pass to the `summarize` method.

    Example::

        >>> from lightning.pytorch import Trainer
        >>> from lightning.pytorch.callbacks import ModelSummary
        >>> trainer = Trainer(callbacks=[ModelSummary(max_depth=1)])

    """

    def __init__(self, max_depth: int = 1, **summarize_kwargs: Any) -> None:
        self._max_depth: int = max_depth
        self._summarize_kwargs: dict[str, Any] = summarize_kwargs

    @override
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._max_depth:
            return

        model_summary = self._summary(trainer, pl_module)
        summary_data = model_summary._get_summary_data()
        total_parameters = model_summary.total_parameters
        trainable_parameters = model_summary.trainable_parameters
        model_size = model_summary.model_size
        total_training_modes = model_summary.total_training_modes

        if trainer.is_global_zero:
            self.summarize(
                summary_data,
                total_parameters,
                trainable_parameters,
                model_size,
                total_training_modes,
                **self._summarize_kwargs,
            )

    def _summary(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Union[DeepSpeedSummary, Summary]:
        from lightning.pytorch.strategies.deepspeed import DeepSpeedStrategy

        if isinstance(trainer.strategy, DeepSpeedStrategy) and trainer.strategy.zero_stage_3:
            return DeepSpeedSummary(pl_module, max_depth=self._max_depth)
        return summarize(pl_module, max_depth=self._max_depth)

    @staticmethod
    def summarize(
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        **summarize_kwargs: Any,
    ) -> None:
        summary_table = _format_summary_table(
            total_parameters,
            trainable_parameters,
            model_size,
            total_training_modes,
            *summary_data,
        )
        _logger.info("\n" + summary_table)

class SaveTestOutputs(L.Callback):
    """
    Callback to save test outputs after the test ends.
    """
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig, run_cfg: RunConfig, output_filepath: str):
        super().__init__()
        self.test_outputs = []
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.run_cfg = run_cfg
        self.output_filepath = output_filepath
        if not self.output_filepath.endswith(".root"):
            raise ValueError("Output filepath doesn't end with .root extension.")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Collect outputs from the test step.
        """
        labels = {k: v.numpy(force=True) for k,v in batch[1].items()}
        self.test_outputs.append(
            {
                "scores": outputs["scores"],
                "labels": labels,
                "weight": outputs["weight"],
                "spectators": outputs.get("spectators", {}),
            }
        )

    def on_test_end(self, trainer, pl_module):
        """
        Save the collected test outputs to a root file.
        """
        if not self.test_outputs:
            _logger.warning("No test outputs collected.")
            return

        all_scores = np.concatenate([o["scores"] for o in self.test_outputs], axis=0)
        all_labels = {k: np.concatenate([o["labels"][k] for o in self.test_outputs], axis=0) for k in self.test_outputs[0]["labels"].keys()}
        all_weights = np.concatenate([o["weight"] for o in self.test_outputs], axis=0)
        all_spectators = {k: np.concatenate([o["spectators"][k] for o in self.test_outputs], axis=0) for k in self.test_outputs[0]["spectators"].keys()}

        _logger.info(f"Saving test outputs: {len(self.test_outputs)} batches collected.")
        self.save_to_root(all_scores, all_labels, all_weights, all_spectators)

    def save_to_root(self, scores, labels, weights, spectators):
        """
        Save the test outputs to a ROOT file.
        """
        output = {}
        if "_class_label" in labels.keys():
            for idx, label_name in enumerate(self.data_cfg.labels["categorical"]):
                output[label_name] = labels["_class_label"] == idx
                output[label_name + "_score"] = scores[:, idx]
            labels.pop("_class_label")

        if len(labels.keys()) > 0:
            if self.model_cfg.model_params.get("num_classes"):
                n_classes = self.model_cfg.model_params["num_classes"]
            else:
                n_classes = len(self.data_cfg.labels.get("categorical",[]))
            for idx, label_name in enumerate(self.data_cfg.labels["continuous"]):
                output[label_name] = labels[label_name]
                output[label_name + "_pred"] = scores[:, n_classes + idx]

        if self.data_cfg.get('weights', {}).get('balance_weights', False):
            output["_balanced_weights"] = weights
        else:
            output["_weights"] = weights
        output.update(spectators)
        write_file(self.output_filepath, ak.Array(output))

        _logger.info(f"Test outputs saved to {self.output_filepath}.")

class SaveONNX(L.Callback):
    """
    Callback to save test outputs after the test ends.
    """
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig, run_cfg: RunConfig, onnx_path: str):
        super().__init__()
        self.test_outputs = []
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.run_cfg = run_cfg
        self.onnx_path = onnx_path
        if not self.onnx_path.endswith(".onnx"):
            raise ValueError("ONNX filepath doesn't end with .onnx extension.")

    def on_test_end(self, trainer, pl_module):
        """
        Save the model to ONNX format after the test ends.
        """
        _logger.info("Saving model to ONNX format...")
        pl_module.eval()
        model = pl_module.to('cpu')

        dummy_input = []

        for feat_name, feat_list in self.data_cfg.inputs.items():
            feat_shape = None
            if feat_name == "evt_feats":
                feat_shape = (1, len(feat_list)) # e.g. (1, 10) for 10 event features
            else:
                feat_shape = (1, len(feat_list), 1) # e.g. (1, 6, 1) for 6 features per particle
            dummy_input.append(torch.randn(feat_shape, dtype=torch.float32))
        dummy_input = tuple(dummy_input)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            self.onnx_path,
            export_params=True,
        )
        _logger.info(f"Model saved to {self.onnx_path}")