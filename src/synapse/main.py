import argparse
import time
import os
import glob
import random
import copy

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from synapse.core.config import ConfigManager
from synapse.core.logger import EnhancedLogger
from synapse.core.callbacks import SaveTestOutputs, SaveONNX
from synapse.core.model_module import ModelModule
from synapse.core.data_module import DataModule
from synapse.third_party.model_summary import ModelSummary


def update_file_path(run_dir, file_path: str, replace_auto: str = "", suffix: str = "") -> str:
    dirname = os.path.dirname(file_path)
    if dirname:
        if os.path.isabs(dirname):
            updated_file_path = file_path
        else:
            updated_file_path = os.path.join(run_dir, file_path)
    else:
        updated_file_path = os.path.join(run_dir, file_path)
    os.makedirs(os.path.dirname(updated_file_path), exist_ok=True)
    if suffix:
        suffix = f"_{suffix}"
    if '{auto}' in updated_file_path:
        if replace_auto == "":
            replace_auto = time.strftime('%Y%m%d_%H%M%S')
        updated_file_path = updated_file_path.replace('{auto}', replace_auto + f'{suffix}')
    return updated_file_path

def train(model, model_config, data_config, run_config,
          train_file_paths, val_file_paths, test_file_paths,
          _logger, run_info_str, path_suffix: str = ""):
    _logger.info(f"{len(train_file_paths)} Train files: {train_file_paths}")
    _logger.info(f"{len(val_file_paths)} Validation files: {val_file_paths}")
    _logger.info(f"{len(test_file_paths)} Test files: {test_file_paths}")

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
        save_dir=run_config.run_dir,
        name=f"TensorBoardLogs_{run_info_str}"
    )

    trainer_callbacks: list[Callback] = [ModelSummary(max_depth=1)]
    best_model_checkpoint_callback = None
    last_checkpoint_callback = None

    if run_config.checkpoint_dir:
        checkpoint_dir = update_file_path(run_config.run_dir, run_config.checkpoint_dir, run_info_str, path_suffix)
        # Optional: save every epoch
        if run_config.save_ckpt_each_epoch:
            each_epoch_checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="model_epoch={epoch}",
                save_top_k=-1,
                every_n_epochs=1,
                save_on_train_epoch_end=False
            )
            trainer_callbacks.append(each_epoch_checkpoint_callback)

        # Optional: keep last checkpoint as a fallback
        last_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="last",
            save_top_k=1,
            save_last=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False
        )
        trainer_callbacks.append(last_checkpoint_callback)

        # save the best checkpoint according to monitored metric
        monitor_metric_name = ''
        monitor_metric_mode = ''
        for metric_name, metric_fn_dict in model.metrics.items():
            if 'val' in metric_fn_dict["stages"] and metric_fn_dict["on_epoch"] and metric_fn_dict["is_monitor"]:
                monitor_metric_name = f"val_{metric_name}_epoch"
                monitor_metric_mode = metric_fn_dict["mode"]
                break
        if monitor_metric_name:
            best_model_checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f"BEST-model-epoch={{epoch}}-{monitor_metric_name}={{{monitor_metric_name}:.4f}}",
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

    do_train = 'train' in run_config.run_mode
    do_test = 'test' in run_config.run_mode
    checkpoint_for_test = None

    if do_train:
        _logger.info("Running in training mode...")
        trainer.fit(model=model, datamodule=data_module)

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
            model_params = copy.deepcopy(model_config.model_params)
            model_params['for_inference'] = True

            model = ModelModule.load_from_checkpoint(
                checkpoint_path=checkpoint_for_test,
                run_cfg=run_config,
                model_class=model_config.model,
                model_params=model_params,
                loss_fn=model_config.loss_function['name'],
                loss_params=model_config.loss_function['params'],
                optimizer=model_config.optimizer,
                start_lr=model_config.start_lr,
                lr_scheduler=model_config.lr_scheduler,
                metrics=model_config.metrics,
                load_model=model_config.load_model,
            )
            _logger.info("Running in test mode using checkpoint: %s", checkpoint_for_test)
        else:
            _logger.info("Running in test mode using in-memory trained model...")

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

    # Initialize the logger
    logger_config = run_config.logger_config

    run_info_str = (f"{time.strftime('%Y%m%d_%H%M%S')}_{model_config.model.split('.')[-1]}"
                    f"_epochs{run_config.epochs}_start{run_config.start_epoch}"
                    f"_bs{run_config.batch_size}_lr{model_config.start_lr}"
                    f"_opt_{model_config.optimizer}_lrSche_{model_config.lr_scheduler}")

    if logger_config.get('log_file'):
        logger_config['log_file'] = update_file_path(run_config.run_dir, logger_config['log_file'], run_info_str, 'info')
    if logger_config.get('debug_file'):
        logger_config['debug_file'] = update_file_path(run_config.run_dir, logger_config['debug_file'], run_info_str, 'debug')

    _logger = EnhancedLogger.from_config(logger_config).get_logger()

    _logger.info("Starting Synapse...")
    if logger_config.get('log_file'):
        _logger.info("Writing logs to file: %s", logger_config['log_file'])
    if logger_config.get('debug_file'):
        _logger.debug("Writing debug logs to file: %s", logger_config['debug_file'])

    if model_config.load_model is None:
        model = ModelModule(
            run_cfg=run_config,
            model_class=model_config.model,
            model_params=model_config.model_params,
            loss_fn=model_config.loss_function['name'],
            loss_params=model_config.loss_function['params'],
            optimizer=model_config.optimizer,
            start_lr=model_config.start_lr,
            lr_scheduler=model_config.lr_scheduler,
            metrics=model_config.metrics,
        )
    else:
        model = ModelModule.load_from_checkpoint(
            checkpoint_path=model_config.load_model,
            run_cfg=run_config,
            model_class=model_config.model,
            model_params=model_config.model_params,
            loss_fn=model_config.loss_function['name'],
            loss_params=model_config.loss_function['params'],
            optimizer=model_config.optimizer,
            start_lr=model_config.start_lr,
            lr_scheduler=model_config.lr_scheduler,
            metrics=model_config.metrics,
            load_model=model_config.load_model,
        )

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
                data_config.train_selection = (f"{base_selection}({cv_var}%{k_folds} != {i}) & "
                                                f"({cv_var}%{k_folds} != {(i+1)%k_folds})")
                data_config.val_selection = f"{base_selection}({cv_var}%{k_folds} == {(i+1)%k_folds})"
                data_config.test_selection = f"{base_selection}({cv_var}%{k_folds} == {i})"
                _logger.info(f"======= Running Fold {i} of {k_folds} =======")
                train(model, model_config, data_config, run_config,
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
                    if f"fold_{(i+k_folds-2)%k_folds}" in file_path:
                        val_file_paths.append(file_path)
                    if f"fold_{(i+k_folds-1)%k_folds}" in file_path:
                        test_file_paths.append(file_path)
                _logger.info(f"======= Running Fold {i} of {k_folds} =======")
                train(model, model_config, data_config, run_config,
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
        train(model, model_config, data_config, run_config,
                train_file_paths, val_file_paths, test_file_paths,
                _logger, run_info_str)
        #TODO: checkpoint loading support
        #TODO: test only mode
        #TODO: resume training support
        #TODO: standalone onnx export support


if __name__ == '__main__':
    main()