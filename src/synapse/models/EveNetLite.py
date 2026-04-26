from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import yaml
from evenet.control.global_config import DotDict
from evenet_lite import EveNetLite
from evenet_lite.classifier import EvenetLiteClassifier
from evenet_lite.hf_utils import load_pretrained_weights


class EveNetLiteWrapper(nn.Module):
    """
    Thin Synapse-compatible wrapper around the installed ``evenet_lite.EveNetLite`` model.

    Synapse calls models with positional inputs derived from ``data.inputs`` order, while
    EveNet-Lite expects keyword inputs ``x``, ``x_mask`` and ``globals``. This wrapper
    resolves that mismatch and also adapts Synapse's ParticleTransformer-style object tensor
    layout ``(batch, features, objects)`` to EveNet-Lite's expected layout
    ``(batch, objects, features)``.
    """

    def __init__(
            self,
            class_labels: Sequence[str],
            global_input_dim: int = 10,
            sequential_input_dim: int = 7,
            config_path: str | None = None,
            pretrained: bool = False,
            pretrained_source: str = "hf",
            pretrained_path: str | None = None,
            pretrained_repo_id: str = EvenetLiteClassifier.DEFAULT_HF_REPO_ID,
            pretrained_filename: str = EvenetLiteClassifier.DEFAULT_HF_REPO_FILENAME,
            pretrained_cache_dir: str | None = None,
            n_ensemble: int = 1,
            ensemble_mode: str = "independent",
            use_adapter: bool = False,
            for_inference: bool = False,
            input_order: Sequence[str] | None = None,
            x_key: str = "x",
            globals_key: str = "globals",
            x_mask_key: str = "x_mask",
    ) -> None:
        super().__init__()

        self.class_labels = list(class_labels)
        self.global_input_dim = global_input_dim
        self.sequential_input_dim = sequential_input_dim
        self.for_inference = for_inference
        self.input_order = list(input_order) if input_order is not None else None
        self.x_key = x_key
        self.globals_key = globals_key
        self.x_mask_key = x_mask_key

        config = self._load_config(config_path)
        self.model = EveNetLite(
            config=config,
            global_input_dim=global_input_dim,
            sequential_input_dim=sequential_input_dim,
            cls_label=self.class_labels,
            n_ensemble=n_ensemble,
            ensemble_mode=ensemble_mode,
            use_adapter=use_adapter,
        )

        if pretrained:
            self._load_pretrained(
                source=pretrained_source,
                local_path=pretrained_path,
                repo_id=pretrained_repo_id,
                filename=pretrained_filename,
                cache_dir=pretrained_cache_dir,
            )

    @staticmethod
    def _default_config_path() -> Path:
        import evenet_lite

        return Path(evenet_lite.__file__).resolve().parent / "config" / "default_network_config.yaml"

    def _load_config(self, config_path: str | None) -> DotDict:
        resolved_path = Path(config_path).expanduser() if config_path else self._default_config_path()
        with open(resolved_path, "r", encoding="utf-8") as f:
            return DotDict(yaml.safe_load(f))

    def _load_pretrained(
            self,
            source: str,
            local_path: str | None,
            repo_id: str | None,
            filename: str | None,
            cache_dir: str | None,
    ) -> None:
        checkpoint: Any = None
        if source == "hf":
            if repo_id is None or filename is None:
                raise ValueError("pretrained_repo_id and pretrained_filename are required when pretrained_source='hf'.")
            checkpoint = load_pretrained_weights(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        elif source == "local":
            if local_path is None:
                raise ValueError("pretrained_path is required when pretrained_source='local'.")
            checkpoint = torch.load(Path(local_path).expanduser(), map_location="cpu")
        else:
            raise ValueError(f"Unsupported pretrained_source: {source}")

        if checkpoint is None:
            raise RuntimeError("Failed to load EveNet-Lite pretrained weights.")

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if state_dict is None and isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model")
        if state_dict is None:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError("Loaded pretrained checkpoint does not contain a valid state_dict.")

        cleaned_state_dict = {
            key.replace("model.", "").replace("module.", ""): value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(cleaned_state_dict, strict=False)

    def _named_inputs_from_args(self, args: tuple[torch.Tensor, ...], named_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        resolved = dict(named_inputs)
        if not args:
            return resolved

        if self.input_order is not None:
            if len(args) != len(self.input_order):
                raise ValueError(
                    f"Expected {len(self.input_order)} positional inputs based on input_order={self.input_order}, got {len(args)}."
                )
            resolved.update(dict(zip(self.input_order, args)))
            return resolved

        default_order = [self.x_key, self.globals_key, self.x_mask_key]
        if len(args) > len(default_order):
            raise ValueError(
                "Received more positional inputs than EveNetLiteWrapper can infer automatically. "
                "Set model_params.input_order to match data.inputs order."
            )
        resolved.update(dict(zip(default_order, args)))
        return resolved

    @staticmethod
    def _coalesce(tensors: dict[str, torch.Tensor], *names: str) -> torch.Tensor | None:
        for name in names:
            if name in tensors:
                return tensors[name]
        return None

    def _prepare_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"EveNetLiteWrapper expects x to be 3D, got shape {tuple(x.shape)}")

        if x.shape[-1] == self.sequential_input_dim:
            return x.float()
        if x.shape[1] == self.sequential_input_dim:
            return x.transpose(1, 2).contiguous().float()

        raise ValueError(
            f"Could not align x with sequential_input_dim={self.sequential_input_dim}. Got shape {tuple(x.shape)}."
        )

    def _prepare_mask(self, x_mask: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        if x_mask is None:
            return torch.ones(x.shape[:2], device=x.device, dtype=torch.float32)

        if x_mask.dim() == 3:
            if x_mask.shape[1] == 1:
                x_mask = x_mask.squeeze(1)
            elif x_mask.shape[-1] == 1:
                x_mask = x_mask.squeeze(-1)
        if x_mask.dim() != 2:
            raise ValueError(f"EveNetLiteWrapper expects x_mask to be 2D after squeezing, got shape {tuple(x_mask.shape)}")

        if x_mask.shape != x.shape[:2]:
            raise ValueError(
                f"x_mask shape {tuple(x_mask.shape)} is incompatible with x shape {tuple(x.shape)}."
            )
        return (x_mask != 0).float()

    def _prepare_globals(self, globals_tensor: torch.Tensor) -> torch.Tensor:
        if globals_tensor.dim() == 1:
            globals_tensor = globals_tensor.unsqueeze(0)
        if globals_tensor.dim() != 2:
            raise ValueError(
                f"EveNetLiteWrapper expects globals to be 2D, got shape {tuple(globals_tensor.shape)}"
            )
        if globals_tensor.shape[-1] != self.global_input_dim:
            raise ValueError(
                f"globals last dimension must match global_input_dim={self.global_input_dim}, got shape {tuple(globals_tensor.shape)}"
            )
        return globals_tensor.float()

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
        tensors = self._named_inputs_from_args(args, kwargs)
        x = self._coalesce(tensors, self.x_key, "x", "obj_feats")
        globals_tensor = self._coalesce(tensors, self.globals_key, "globals", "evt_feats")
        x_mask = self._coalesce(tensors, self.x_mask_key, "x_mask", "mask")

        if x is None:
            raise ValueError("Missing EveNetLite input tensor 'x'.")
        if globals_tensor is None:
            raise ValueError("Missing EveNetLite input tensor 'globals'.")

        x = self._prepare_x(x)
        globals_tensor = self._prepare_globals(globals_tensor)
        x_mask = self._prepare_mask(x_mask, x)

        return self.model(x=x, x_mask=x_mask, globals=globals_tensor)

