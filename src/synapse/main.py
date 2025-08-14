import argparse
import time
import os
import glob

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from synapse.core.config import ConfigManager
from synapse.core.logger import EnhancedLogger
from synapse.core.callbacks import SaveTestOutputs, SaveONNX
from synapse.core.model_module import ModelModule
from synapse.core.data_module import DataModule


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

    deterministic = False
    if run_config.seed:
        L.seed_everything(run_config.seed, workers=True, verbose=False)
        _logger.info(f"Set random seed to {run_config.seed}")
        deterministic = True
    else:
        _logger.info("No random seed specified")

    data_module = DataModule(
        data_cfg=data_config,
        run_cfg=run_config,
        train_file_list=train_file_paths,
        val_file_list=val_file_paths,
        test_file_list=test_file_paths
    )

    tb_logger = TensorBoardLogger(save_dir=run_config.run_dir, name=f"TensorBoardLogs_{run_info_str}")

    trainer_callbacks = []
    if run_config.get("test_output"):
        test_output = update_file_path(run_config.run_dir, run_config.test_output, run_info_str, path_suffix)
        test_output_callback = SaveTestOutputs(
            data_cfg=data_config,
            model_cfg=model_config,
            run_cfg=run_config,
            output_filepath=test_output
        )
        trainer_callbacks.append(test_output_callback)

    if run_config.get("onnx_path"):
        onnx_path = update_file_path(run_config.run_dir, run_config.onnx_path, run_info_str, path_suffix)
        onnx_callback = SaveONNX(
            data_cfg=data_config,
            model_cfg=model_config,
            run_cfg=run_config,
            onnx_path=onnx_path
        )
        trainer_callbacks.append(onnx_callback)
    # TODO: customized checkpoint callback (save ckpt to another place, not tb logger dir)
    # TODO: model_summary callback.

    trainer = L.Trainer(
        accelerator=run_config.device,
        devices=run_config.n_devices,
        deterministic=deterministic,
        num_sanity_val_steps=2 if run_config.val_sanity_check else 0,
        default_root_dir=run_config.run_dir,
        enable_checkpointing=True,
        max_epochs=run_config.epochs,
        logger=tb_logger,
        precision= '16-mixed' if run_config.use_amp else '32-true',
        enable_progress_bar=True,
        enable_model_summary=True,
        inference_mode=True,
        callbacks= trainer_callbacks,
    )

    if 'train' in run_config.run_mode:
        _logger.info("Running in training mode...")
        data_module.setup('fit')
        trainer.fit(model=model, datamodule=data_module)
    if 'test' in run_config.run_mode:
        _logger.info("Running in test mode...")
        trainer.test(model=model, datamodule=data_module)

def main():
    parser = argparse.ArgumentParser(description="Run Synapse")
    parser.add_argument('-d', '--data_config', type=str, required=True, help='Data configuration file path')
    parser.add_argument('-m', '--model_config', type=str, required=True, help='Model configuration file path')
    parser.add_argument('-r', '--run_config', type=str, required=True, help='Run configuration file path')

    args = parser.parse_args()
    
    config_manager = ConfigManager(
        data_cfg_file=args.data_config,
        model_cfg_file=args.model_config,
        run_cfg_file=args.run_config
    )
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

    if run_config.cross_validation:
        # TODO: modify the model checkpoint name, etc., according to fold number
        _logger.info("Cross validation: ON")
        if run_config.k_folds is None:
            raise ValueError("k_folds is not specified in run configuration, cannot perform cross-validation.")
        k_folds = run_config.k_folds
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
            for i in range(k_folds):
                data_config.train_selection = (f"{base_selection}({cv_var}%{k_folds} != {i}) & "
                                                f"({cv_var}%{k_folds} != {(i+1)%k_folds})")
                data_config.val_selection = f"{base_selection}({cv_var}%{k_folds} == {(i+1)%k_folds})"
                data_config.test_selection = f"{base_selection}({cv_var}%{k_folds} == {i})"
                _logger.info(f"======= Running Fold {i} of {k_folds} =======")
                train(model, model_config, data_config, run_config,
                        file_paths, file_paths, file_paths,
                        _logger, run_info_str, "fold_{i}")
        else: # very inflexible way, if no cross-validation variable is specified.
            _logger.info("No cross-validation variable specified.")
            _logger.info("Checking if all folds ('fold_X' in file name) are present in the dataset...")
            for i in range(k_folds):
                if sum(f"fold_{i}" in file_path for file_path in file_paths) == 0:
                    raise RuntimeError(f"No file found for fold {i}")
            # Create a list of file paths for each fold
            for i in range(k_folds):
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
                        _logger, run_info_str, "fold_{i}")
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


if __name__ == '__main__':
    main()