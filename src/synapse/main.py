import argparse
import time
import os
import glob
import random
import copy

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from synapse.core.config import ConfigManager
from synapse.core.logger import EnhancedLogger
from synapse.core.callbacks import SaveTestOutputs, SaveONNX, StageScopedProgressBar
from synapse.core.model_module import ModelModule
from synapse.core.data_module import DataModule


def resolve_run_path(run_dir: str, file_path: str) -> str:
    dirname = os.path.dirname(file_path)
    if dirname:
        if os.path.isabs(dirname):
            return file_path
        return os.path.join(run_dir, file_path)
    return os.path.join(run_dir, file_path)


def update_file_path(run_dir, file_path: str, replace_auto: str = "", suffix: str = "") -> str:
    updated_file_path = resolve_run_path(run_dir, file_path)
    if suffix:
        suffix = f"_{suffix}"
    if '{auto}' in updated_file_path:
        if replace_auto == "":
            replace_auto = time.strftime('%Y%m%d_%H%M%S')
        updated_file_path = updated_file_path.replace('{auto}', replace_auto + f'{suffix}')
    os.makedirs(os.path.dirname(updated_file_path), exist_ok=True)
    return updated_file_path

def build_model_module(model_config, run_config, checkpoint_path: str | None = None, for_inference: bool = False):
    model_params = copy.deepcopy(model_config.model_params)
    if for_inference:
        model_params['for_inference'] = True

    model_kwargs = dict(
        run_cfg=run_config,
        model_class=model_config.model,
        model_params=model_params,
        loss_fn=model_config.loss_function['name'],
        loss_params=model_config.loss_function['params'],
        training_step_func=model_config.get('training_step_function') or "default",
        validation_step_func=model_config.get('validation_step_function') or "default",
        test_step_func=model_config.get('test_step_function') or "default",
        optimizer=model_config.optimizer,
        start_lr=model_config.start_lr,
        lr_scheduler=model_config.get('lr_scheduler'),
        metrics=model_config.get('metrics'),
    )

    if checkpoint_path:
        return ModelModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **model_kwargs,
        )

    return ModelModule(**model_kwargs)


def get_checkpoint_epoch(checkpoint_path: str) -> int:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_epoch = checkpoint.get('epoch')
    if not isinstance(checkpoint_epoch, int):
        raise ValueError(f"Checkpoint does not contain a valid integer epoch: {checkpoint_path}")
    return checkpoint_epoch


def resolve_resume_checkpoint(run_config, model_config, _logger, path_suffix: str = "") -> tuple[str | None, str | None]:
    if run_config.start_epoch == 0:
        return None, None

    if run_config.start_epoch < 0:
        raise ValueError(f"`run_cfg.start_epoch` must be >= 0, got {run_config.start_epoch}")

    if run_config.epochs <= run_config.start_epoch:
        raise ValueError(
            "Resumed training requires `run_cfg.epochs` to be greater than `run_cfg.start_epoch`. "
            f"Got `epochs`={run_config.epochs}, `start_epoch`={run_config.start_epoch}."
        )

    expected_checkpoint_epoch = run_config.start_epoch - 1

    load_model_path = model_config.get('load_model')
    if load_model_path:
        checkpoint_path = load_model_path
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint from `model_cfg.load_model` was not found: {checkpoint_path}")

        checkpoint_epoch = get_checkpoint_epoch(checkpoint_path)
        if checkpoint_epoch != expected_checkpoint_epoch:
            raise ValueError(
                "Configured `run_cfg.start_epoch` does not match the checkpoint state from `model_cfg.load_model`. "
                f"Expected checkpoint epoch '{expected_checkpoint_epoch}', got '{checkpoint_epoch}' from: {checkpoint_path}."
            )

        return checkpoint_path, None

    if not run_config.checkpoint_dir:
        raise ValueError(
            "Resumed training requires either model.load_model or run.checkpoint_dir to locate a checkpoint."
        )

    checkpoint_dir_pattern = resolve_run_path(run_config.run_dir, run_config.checkpoint_dir)
    if '{auto}' in checkpoint_dir_pattern:
        auto_pattern = f"*_{path_suffix}" if path_suffix else "*"
        checkpoint_dir_pattern = checkpoint_dir_pattern.replace('{auto}', auto_pattern)

    checkpoint_candidates = []
    for matched_path in glob.glob(checkpoint_dir_pattern):
        if os.path.isdir(matched_path):
            checkpoint_candidates.extend(
                glob.glob(os.path.join(matched_path, '**', '*.ckpt'), recursive=True)
            )
        elif matched_path.endswith('.ckpt') and os.path.isfile(matched_path):
            checkpoint_candidates.append(matched_path)

    if not checkpoint_candidates:
        raise FileNotFoundError(
            "Automatic resume checkpoint discovery failed. "
            f"No checkpoint files were found under: {checkpoint_dir_pattern}"
        )

    matched_checkpoints = []
    for checkpoint_path in sorted(set(checkpoint_candidates)):
        try:
            checkpoint_epoch = get_checkpoint_epoch(checkpoint_path)
        except Exception as exc:
            _logger.debug("Skipping unreadable checkpoint %s: %s", checkpoint_path, exc)
            continue

        if checkpoint_epoch == expected_checkpoint_epoch:
            matched_checkpoints.append(checkpoint_path)

    if not matched_checkpoints:
        raise FileNotFoundError(
            "Automatic resume checkpoint discovery failed. "
            f"No checkpoint matching epoch {expected_checkpoint_epoch} was found under: {checkpoint_dir_pattern}"
        )

    matched_checkpoints.sort(key=os.path.getmtime, reverse=True)
    checkpoint_path = matched_checkpoints[0]
    if len(matched_checkpoints) > 1:
        _logger.info(
            "Multiple checkpoints matched resume epoch %d; using the most recently modified one: %s",
            expected_checkpoint_epoch,
            checkpoint_path,
        )

    return checkpoint_path, os.path.dirname(checkpoint_path)


def train(model_config, data_config, run_config,
          train_file_paths, val_file_paths, test_file_paths,
          _logger, run_info_str, path_suffix: str = ""):
    # explicitly set random seed, either by user or automatically
    if run_config.seed:
        _logger.info(f"Set random seed to {run_config.seed}")
        L.seed_everything(run_config.seed, workers=True, verbose=False)
    else:
        rnd_seed = random.randint(1, 65536)
        _logger.info("No random seed specified")
        _logger.info(f"Set auto generated random seed to {rnd_seed}")
        L.seed_everything(rnd_seed, workers=True, verbose=False)

    deterministic = True

    data_module = DataModule(
        data_cfg=data_config,
        run_cfg=run_config,
        train_file_list=train_file_paths,
        val_file_list=val_file_paths,
        test_file_list=test_file_paths
    )

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(run_config.run_dir, "TensorBoardLogs"),
        name=f"TensorBoardLogs_{run_info_str}"
    )

    do_train = 'train' in run_config.run_mode
    do_test = 'test' in run_config.run_mode
    resume_checkpoint_path = None
    resume_checkpoint_dir = None

    if do_train:
        resume_checkpoint_path, resume_checkpoint_dir = resolve_resume_checkpoint(
            run_config=run_config,
            model_config=model_config,
            _logger=_logger,
            path_suffix=path_suffix,
        )

    if resume_checkpoint_path:
        remaining_epochs = run_config.epochs - run_config.start_epoch
        _logger.info("Resuming training from checkpoint '%s'.", resume_checkpoint_path)
        _logger.info(
            "Next epoch: %d, target max epoch: %d, remaining epochs to run: %d.",
            run_config.start_epoch,
            run_config.epochs,
            remaining_epochs,
        )
        model = build_model_module(model_config=model_config, run_config=run_config)
    elif model_config.get('load_model') and do_train:
        _logger.info(
            "Initializing training model weights from checkpoint without resuming optimizer/scheduler state: %s",
            model_config.get('load_model'),
        )
        model = build_model_module(
            model_config=model_config,
            run_config=run_config,
            checkpoint_path=model_config.get('load_model'),
        )
    else:
        model = build_model_module(model_config=model_config, run_config=run_config)

    trainer_callbacks: list[Callback] = [
        StageScopedProgressBar(),
        ModelSummary(max_depth=1),
        LearningRateMonitor(logging_interval='step')
    ]
    best_model_checkpoint_callback = None
    last_checkpoint_callback = None

    if run_config.checkpoint_dir:
        if resume_checkpoint_dir and model_config.get('load_model') is None:
            checkpoint_dir = resume_checkpoint_dir
        else:
            checkpoint_dir = update_file_path(run_config.run_dir, run_config.checkpoint_dir, run_info_str, path_suffix)

        if run_config.save_ckpt_each_epoch:
            # Save every epoch and also keep the canonical last checkpoint.
            # This avoids creating two stateful ModelCheckpoint callbacks with the
            # same state key (which Lightning does not allow when checkpointing
            # callback state for resume).
            each_epoch_checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="model_{epoch}",
                save_top_k=-1,
                save_last=True,
                every_n_epochs=1,
                save_on_train_epoch_end=False
            )
            trainer_callbacks.append(each_epoch_checkpoint_callback)
            last_checkpoint_callback = each_epoch_checkpoint_callback
        else:
            # Keep only the latest checkpoint as a fallback when per-epoch saving
            # is disabled.
            last_checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="last_{epoch}",
                save_top_k=1,
                save_last=True,
                every_n_epochs=1,
                save_on_train_epoch_end=False
            )
            trainer_callbacks.append(last_checkpoint_callback)

        # Save the best checkpoint according to validation loss (default) or monitored metric
        monitor_metric_name = 'val_loss_epoch'
        monitor_metric_mode = 'min'
        for metric_name, metric_fn_dict in model.metrics.items():
            if 'val' in metric_fn_dict["stages"] and metric_fn_dict["on_epoch"] and metric_fn_dict["is_monitor"]:
                monitor_metric_name = f"val_{metric_name}_epoch"
                monitor_metric_mode = metric_fn_dict["mode"]
                break
        best_model_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"BEST-model-{{epoch}}-{{{monitor_metric_name}:.4f}}",
            monitor=monitor_metric_name,
            mode=monitor_metric_mode,
            save_top_k=1,
            every_n_epochs=1,
            save_on_train_epoch_end=False
        )
        trainer_callbacks.append(best_model_checkpoint_callback)

    if run_config.test_output:
        test_output = update_file_path(run_config.run_dir, run_config.test_output, run_info_str, path_suffix)
        trainer_callbacks.append(
            SaveTestOutputs(
                data_cfg=data_config,
                model_cfg=model_config,
                run_cfg=run_config,
                output_filepath=test_output
            )
        )

    if run_config.onnx_path:
        onnx_path = update_file_path(run_config.run_dir, run_config.onnx_path, run_info_str, path_suffix)
        trainer_callbacks.append(
            SaveONNX(
                data_cfg=data_config,
                model_cfg=model_config,
                run_cfg=run_config,
                onnx_path=onnx_path
            )
        )

    trainer = L.Trainer(
        accelerator=run_config.device,
        devices=run_config.n_devices,
        deterministic=deterministic,
        num_sanity_val_steps=2 if run_config.val_sanity_check else 0,
        default_root_dir=run_config.run_dir,
        enable_checkpointing=bool(run_config.checkpoint_dir),
        max_epochs=run_config.epochs,
        logger=tb_logger,
        precision= '16-mixed' if run_config.use_amp else '32-true',
        enable_progress_bar=True,
        enable_model_summary=False,
        inference_mode=True,
        callbacks= trainer_callbacks,
    )
    checkpoint_for_test = None

    if do_train:
        _logger.info("Running in training mode...")
        _logger.info("%d Train files:\n  %s", len(train_file_paths),
                    "\n  ".join(train_file_paths) if train_file_paths else "<none>")
        _logger.info("%d Validation files:\n  %s", len(val_file_paths),
                    "\n  ".join(val_file_paths) if val_file_paths else "<none>")
        if data_config.get("train_selection"):
            _logger.info("Train entry selection: %s", data_config.train_selection)
        else:
            _logger.info("Train entry selection: %s", data_config.selection)
        if data_config.get("val_selection"):
            _logger.info("Validation entry selection: %s", data_config.val_selection)
        else:
            _logger.info("Validation entry selection: %s", data_config.selection)

        trainer.fit(model=model, datamodule=data_module, ckpt_path=resume_checkpoint_path)

        if best_model_checkpoint_callback and best_model_checkpoint_callback.best_model_path:
            checkpoint_for_test = best_model_checkpoint_callback.best_model_path
            _logger.info("Using best checkpoint for test: %s", checkpoint_for_test)
        elif last_checkpoint_callback and last_checkpoint_callback.last_model_path:
            checkpoint_for_test = last_checkpoint_callback.last_model_path
            _logger.info("No best checkpoint available, using last checkpoint for test: %s", checkpoint_for_test)
        else:
            _logger.info("No checkpoint available after training; test will use in-memory model.")

    if do_test:
        if not do_train:
            checkpoint_for_test = model_config.load_model
            if not checkpoint_for_test:
                raise ValueError(
                    "Test-only mode requires `model.load_model` to be set in the configuration."
                )

        if checkpoint_for_test:
            model = build_model_module(
                model_config=model_config,
                run_config=run_config,
                checkpoint_path=checkpoint_for_test,
                for_inference=True,
            )
            _logger.info("Running in test mode using checkpoint: %s", checkpoint_for_test)
        else:
            _logger.info("Running in test mode using in-memory trained model...")

        _logger.info("%d Test files:\n  %s", len(test_file_paths),
                    "\n  ".join(test_file_paths) if test_file_paths else "<none>")
        if data_config.get("test_selection"):
            _logger.info("Test entry selection: %s", data_config.test_selection)
        else:
            _logger.info("Test entry selection: %s", data_config.selection)

        trainer.test(model=model, datamodule=data_module)

def main():
    parser = argparse.ArgumentParser(description="Run Synapse")
    parser.add_argument('-c', '--config', type=str, required=True, help='Configuration file path')

    args = parser.parse_args()
    
    config_manager = ConfigManager(cfg_file_path=args.config)

    data_config = config_manager.data
    model_config = config_manager.model
    run_config = config_manager.run

    os.makedirs(run_config.run_dir, exist_ok=True)

    run_info_str = (f"{time.strftime('%Y%m%d_%H%M%S')}_{model_config.model.split('.')[-1]}"
                    f"_epochs{run_config.epochs}_start{run_config.start_epoch}"
                    f"_bs{run_config.batch_size}_lr{model_config.start_lr}"
                    f"_opt_{model_config.optimizer}_lrSche_{model_config.lr_scheduler}")

    # Initialize the logger
    logger_config = run_config.logger_config

    if logger_config.get('log_file'):
        logger_config['log_file'] = update_file_path(run_config.run_dir, logger_config['log_file'], run_info_str, 'info')
    if logger_config.get('debug_file'):
        logger_config['debug_file'] = update_file_path(run_config.run_dir, logger_config['debug_file'], run_info_str, 'debug')

    enhanced_logger = EnhancedLogger.from_config(logger_config)
    bridged_loggers = logger_config.get('bridge_loggers',[])
    if isinstance(bridged_loggers, str):
        bridged_loggers = [bridged_loggers]
    else:
        bridged_loggers = list(bridged_loggers)
    if bridged_loggers:
        enhanced_logger.bridge_loggers(
            bridged_loggers,
            propagate=logger_config.get('bridge_logger_propagate', False)
        )
    _logger = enhanced_logger.get_logger()

    _logger.info("Starting Synapse...")
    if logger_config.get('log_file'):
        _logger.info("Writing logs to file: %s", logger_config['log_file'])
    if logger_config.get('debug_file'):
        _logger.debug("Writing debug logs to file: %s", logger_config['debug_file'])

    if run_config.cross_validation:
        _logger.info("Cross validation: ON")
        if run_config.k_folds is None:
            raise ValueError("k_folds is not specified in run configuration, cannot perform cross-validation.")
        k_folds = run_config.k_folds
        run_folds = run_config.run_folds
        if run_folds is None:
            run_folds = list(range(run_config.k_folds))
        _logger.info("Cross-validation with %d folds.", k_folds)
        _logger.info("Train/Val/Test files will be merged into a single list then split into folds...")
        file_paths = []
        file_paths.extend(data_config.train_files)
        file_paths.extend(data_config.val_files)
        file_paths.extend(data_config.test_files)
        file_paths = [filepath for path_pattern in file_paths for filepath in glob.glob(path_pattern)]
        if run_config.cross_validation_var:
            cv_var = run_config.cross_validation_var
            _logger.info(f"Cross-validation variable specified: {cv_var}")
            if data_config.selection:
                base_selection = f"({data_config.selection}) & "
            else:
                base_selection = ""
            for i in run_folds:
                data_config.train_selection = (f"{base_selection}({cv_var}%{k_folds} != {(i-2)%k_folds}) & "
                                                f"({cv_var}%{k_folds} != {(i-1)%k_folds})")
                data_config.val_selection = f"{base_selection}({cv_var}%{k_folds} == {(i-2)%k_folds})"
                data_config.test_selection = f"{base_selection}({cv_var}%{k_folds} == {(i-1)%k_folds})"
                _logger.info(f"======= Running Fold {i} of {k_folds} =======")
                train(model_config, data_config, run_config,
                        file_paths, file_paths, file_paths,
                        _logger, run_info_str, f"{k_folds}fold_{i}")
        else: # very inflexible way, if no cross-validation variable is specified.
            _logger.info("No cross-validation variable specified.")
            _logger.info("Checking if all folds ('fold_X' in file name) are present in the dataset...")
            for i in range(k_folds):
                if sum(f"{k_folds}fold_{i}" in file_path for file_path in file_paths) == 0:
                    raise RuntimeError(f"No file found for fold {i}")
            # Create a list of file paths for each fold
            for i in run_folds:
                train_file_paths = []
                val_file_paths = []
                test_file_paths = []
                for file_path in file_paths:
                    for j in range(k_folds-2):
                        if f"fold_{(i+j)%k_folds}" in file_path:
                            train_file_paths.append(file_path)
                    if f"fold_{(i-2)%k_folds}" in file_path:
                        val_file_paths.append(file_path)
                    if f"fold_{(i-1)%k_folds}" in file_path:
                        test_file_paths.append(file_path)
                _logger.info(f"======= Running Fold {i} of {k_folds} =======")
                train(model_config, data_config, run_config,
                        train_file_paths, val_file_paths, test_file_paths,
                        _logger, run_info_str, f"{k_folds}fold_{i}")
    else:
        train_file_paths = []
        val_file_paths = []
        test_file_paths = []
        for file_path in data_config.train_files:
            train_file_paths.extend(glob.glob(file_path))
        for file_path in data_config.val_files:
            val_file_paths.extend(glob.glob(file_path))
        for file_path in data_config.test_files:
            test_file_paths.extend(glob.glob(file_path))
        train(model_config, data_config, run_config,
                train_file_paths, val_file_paths, test_file_paths,
                _logger, run_info_str)
        #TODO: standalone onnx export support


if __name__ == '__main__':
    main()