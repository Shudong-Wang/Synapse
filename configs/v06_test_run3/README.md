# Six-jet fold-0 ParT training

The default student learns from 75% archived EveNet probability and 25% truth. The three output classes, in order, are signal (SM HH), background, and external-conversion background. Each event contains three leptons, up to six central jets, and MET.

## Run on a remote GPU

From the Synapse checkout, activate the Python 3.11 `ML` environment with CUDA-enabled PyTorch and the dependencies in `pyproject.toml`. Copy the prepared `3fold_2.root`, `3fold_1.root`, and `3fold_0.root` files to a directory on the remote machine. The launcher imports Synapse from the checkout.

```bash
conda activate ML
configs/v06_test_run3/train.sh --data-dir /path/to/dataset
```

Placing the files under `configs/v06_test_run3/dataset/` also enables launching `train.sh` without arguments. The data and training outputs are outside version control. The default run uses one GPU, 20 epochs, batch size 512, and seed 42.

If the remote machine already has the original v06 `sr.root` and `extconvcr.root`, you can build the same folds there.

```bash
python configs/v06_test_run3/preprocess.py --input-dir /path/to/v06/ntuples --output /path/to/dataset
```

For the six-jet truth-only baseline, use the same files and settings.

```bash
configs/v06_test_run3/train.sh --data-dir /path/to/dataset --teacher-fraction 0
```

## Training and model selection

The fold files are physical splits by `eventNumber % 3`. Training uses remainder 2, validation uses remainder 1, and final testing uses remainder 0.

For valid teacher scores, the cross-entropy target is `0.75 * [evenet_pS, evenet_pB, evenet_pConv] + 0.25 * one_hot(class_id)`. Events with `evenet_valid == 0` use their full truth label. Each event receives its original absolute weight times the balance factor for its truth class. The loss uses the mean balanced event weight of the training fold for normalization. Only the training objective uses the teacher. Validation and test loss always use truth.

Each fit writes its resolved configurations, log, TensorBoard records, and checkpoints under `output/fold_0/evenet_0.75/fit-*/` or `output/fold_0/truth_only/fit-*/`. Use `--output-dir` to choose another base directory. `validation_scores.root` contains predictions from the epoch with the lowest validation loss. Compare the runs on validation with `n_jet >= 1 && p_extconv_bkg < 0.8` and use `p_signal` as the fit observable. The score trees include the native event identifiers, physics weights, archived EveNet probabilities, and `xgb_score`.

After choosing the model on validation, score the held-out fold by passing that run's `BEST-*.ckpt` file. The launcher uses the saved model configuration beside the checkpoint directory when available. A standalone checkpoint uses the architecture in this recipe.

```bash
configs/v06_test_run3/train.sh --data-dir /path/to/dataset \
    --test-checkpoint /path/to/fit-directory/checkpoints/BEST-*.ckpt
```

The launcher writes `test_scores.root` into a separate `test-*/` directory and prints its location.
