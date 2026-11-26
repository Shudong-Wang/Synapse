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

To extend a completed fit, pass the checkpoint for the last completed epoch. Lightning restores the model, Adam moments and training counters. The example starts a new cosine schedule at `0.0001` and decays it to zero over the next 15 epochs. The extension saves its checkpoints and validation predictions in a separate directory.

```bash
configs/v06_test_run3/train.sh --data-dir /path/to/dataset \
    --extend-checkpoint /path/to/fit-directory/checkpoints/model-epoch=epoch=19-*.ckpt \
    --extra-epochs 15 --restart-lr 0.0001
```

## Class balance for the next Run 3 lepton-origin fit

The next fit with the lepton-origin recipe uses `balance_origin_classes: true` under `loss_function.params`. At fit setup Synapse measures each head’s positive and negative support on the selected training fold. Each valid object receives its event training weight divided by the number of valid objects. Real jets and MET contribute negative labels for the Higgs, top and W/Z/diboson heads.

For a head with effective positive fraction `p`, a positive target gets a BCE multiplier of `0.5 / p` and a negative target gets `0.5 / (1 − p)`. This assigns half the total weight to each class and preserves the total weight of each head. The training log records both multipliers and the checkpoint stores them. Conversion uses its existing lepton-only loss.

The dated Run 3 recipe also attaches two independent event classifiers to the shared event representation. The primary three-class head learns MC truth on every training event and supplies the analysis scores. A separate three-class head matches valid EveNet probabilities with a KL loss coefficient of 0.25. The eight-way softmax head learns the parent process with a cross-entropy coefficient of 0.10, in the order HH, WZ, other VV, VVV, V+jets, Vγ, top and single Higgs. A conversion retains its generator process for this head.

The launcher resolves process labels from the fold dataset’s `summary.json` and records the DSID map in the fit configuration. The process loss weights an event of class `c` by `|w| × N_train / (8 × Σ_c |w|)`, using only the training fold to calculate the denominators. Each process receives one eighth of the total auxiliary weight. Validation loss uses the primary truth head. Scoring exports the main probabilities, the three `p_teacher_*` probabilities and the eight `p_process_*` probabilities. Set `process_loss_weight` to zero for a comparison using the same main, EveNet and object losses.

## Background-rejection plots

Run the comparison on a saved score tree from the Synapse checkout.

```bash
source ./setup.sh
python configs/v06_test_run3/plot_background_rejection.py \
    --scores /path/to/test_scores.root --out /path/to/evaluation/background_rejection
```

The same command accepts `validation_scores.root`. It reads the per-part `summary.json` beside the fold files listed in the saved `data_config.yaml`. Use `--catalog PATH` to supply that manifest explicitly or to use a historical CSV with columns `dsid,analysis_category`.

`rejection_by_background.pdf` compares Synapse, EveNet and XGBoost for inclusive background and each populated parent process. Both pages use positive nominal MC weights and at least one jet. The first measures signal and background efficiencies inside each method’s signal region. The second includes the conversion veto in both efficiencies. Synapse uses its own conversion probability below 0.8. EveNet and XGBoost use valid EveNet predictions with the EveNet conversion probability below 0.8.

`roc_curves.csv` stores the plotted curves from `hhml_eval` and the ratios at matched signal efficiency. `working_points.csv` gives the actual inclusive score cuts, achieved efficiencies, weighted background yields, rejections and ratios for reachable targets from 10% to 95%. The whiskers at 50% and 70% propagate background MC statistics at fixed signal cuts and account for the events shared between methods.

`trafo60_last_three_bins.pdf` shows the expected background yield, background composition and SM HH efficiency in each of the highest three native Trafo60 bins. Each classifier uses the score, selection and edges saved by `hhml_eval` for that fold, including both XGBoost signal regions. The background yield sums signed nominal MC weights over all background processes in the bin. A component’s fraction is its signed MC yield divided by that total. Signal efficiency is the signed signal yield in that bin divided by all SM HH in the corresponding signal region. The plot also gives the background yield and signal efficiency summed over the three bins. The companion CSV contains the exact edges, yields and fractions.

The plotting command finds the native benchmark under `evaluation/hhml_eval/held_out` or `evaluation/hhml_eval/validation` in the production workflow. For a different layout, pass `--benchmark /path/to/matching/fold` containing `benchmark.yaml` and `sig_with_trafo.json`. The script checks the signal and total background in every bin against those histograms before publishing the plots.
