# P2PNet + MovingDroneCrowd++: Colab/VS Code guide

This workflow keeps VS Code and Git as the source of truth, runs GPU work in a
remote Colab kernel, stages the dataset on Colab's fast `/content` disk, and
stores checkpoints/results persistently in Google Drive.

## What is already configured

- Dataset: MovingDroneCrowd++ `train.txt`, `val.txt`, and `test.txt` protocol.
- Labels: each head box `(x, y, w, h)` is converted in memory to
  `(x + w/2, y + h/2)`. The source CSV files are never changed.
- Training: random square patches, scaling, flipping, periodic validation,
  complete resumable checkpoints, TensorBoard, and JSON run metadata.
- Evaluation: frame-count MAE and RMSE with per-frame CSV results.
- Inference: one image or a whole directory, point JSON, count CSV, and marked
  images.
- Large images: overlap-aware tiled inference. Each overlap region has a
  single owner tile, preventing simple double-counting at tile boundaries.

These are **frame-level** results. They are not MDC video individual-counting,
identity, inflow/outflow, or tracking metrics.

## 1. One-time local preparation

### 1.1 Put the modified code in Git

The VS Code Colab extension executes notebook Python on a remote machine. That
remote kernel cannot see ordinary files on your Windows disk. Commit this
repository and push it to a GitHub repository/fork, then put that URL and branch
in `colab_runner.ipynb`.

Typical commands from this repository are:

```powershell
git status
git add .
git commit -m "Add MovingDroneCrowd Colab workflow"
git push -u <your-remote-name> <your-branch-name>
```

Do not commit the 11 GB dataset or training outputs.

### 1.2 Put the dataset in Google Drive

The local dataset currently occupies about 10.96 GiB and contains 7,197 image
frames. A plain `.tar` archive is recommended: JPEG data will not compress much,
and `.tar` avoids thousands of slow Drive reads.

Choose an output path with at least 11 GB free. From PowerShell:

```powershell
tar.exe -cf "D:\TEMP\MovingDroneCrowd++.tar" `
  -C "C:\Users\GabrielSchmitz\Downloads\UFRJ\00_IC\90_Before\05_VisDrone_Dataset\MovingDroneCrowd" `
  "MovingDroneCrowd++"
```

Upload the archive to:

```text
My Drive/P2PNet_MDC/data/MovingDroneCrowd++.tar
```

You may choose another Drive location; update `DATASET_ARCHIVE` in the notebook.
Colab temporarily needs roughly 23 GB free while it holds both the archive and
the extracted dataset. The notebook deletes the `/content` archive copy after
extraction. If the runtime does not have enough disk, use the notebook's
`drive_folder` mode, which is slower, or make a smaller split-specific archive.

### 1.3 Persistent Drive layout

Recommended layout:

```text
My Drive/P2PNet_MDC/
├── data/
│   └── MovingDroneCrowd++.tar
├── external_weights/
└── runs/
    └── <run name>/
        ├── checkpoints/
        ├── tensorboard/
        ├── run_config.json
        ├── training_history.csv
        ├── training_summary.json
        └── history.jsonl
```

The official `weights/SHTechA.pth` checkpoint is already in this repository.
It is useful for checking inference and as an MDC fine-tuning initialization,
but its ShanghaiTech predictions are not valid MDC benchmark results until the
model has been trained/fine-tuned on MDC.

## 2. Open and configure the notebook

Open `colab_runner.ipynb` in VS Code, select the Google Colab GPU kernel, and
use the control panel at the top. The main selector is:

```python
ACTION = "inspect"  # inspect | smoke | benchmark | train | resume | evaluate | inference
```

Choose the initialization, run name, checkpoint, data split, and parameters in
that same cell. Then run setup/staging, read the resolved preflight table, and
run the single action cell. `inspect` is the safe default and does no model
work. In Google Colab the `# @param` annotations may render as form controls;
in the VS Code Colab extension they remain a clearly grouped Python form.

After a runtime reset, repeat setup and dataset staging. Checkpoints in Drive
survive the reset. Local changes to this repository do not reach Colab until
they are committed and pushed to the `REPO_URL`/`REPO_REF` selected above.

The notebook deliberately separates automation from decisions. It automates
repetitive work (commands, validation, checkpointing, timing, and plots), but
the preflight table always shows the resolved inputs before an action runs.
The user still chooses the action, initialization, crop/patch configuration,
checkpoint, split, and final threshold.

## 3. Recommended action order

Use this order before and during the main run:

1. Runtime/GPU report.
2. Repository checkout and dependency installation.
3. Dataset staging and `validate_mdc.py`.
4. One annotated sample visualization.
5. Keep `ACTION="inspect"`, run the preflight, and read every resolved choice.
6. Set `ACTION="smoke"`, rerun the preflight, and run the action. This checks one
   training update, one validation frame, checkpoint creation, timing, and GPU
   memory reporting with the currently selected configuration.
7. Set `ACTION="benchmark"`. It runs short, separate technical trials of
    `512 x 512, 1 patch` and `256 x 256, 4 patches`, then displays measured
    time/memory and rough duration estimates. It does **not** decide which is
    more accurate.
8. Review the benchmark and choose `CROP_SIZE`/`NUM_PATCHES`. Keep the larger
    crop when it fits comfortably and its runtime is acceptable; otherwise use
    the smaller multi-patch option.
9. Resolve initialization and weight-decay decisions, then use
    `ACTION="train"` for the main run.
10. During or after training, rerun the dashboard cell. Use `ACTION="resume"`
    after a runtime interruption, `ACTION="evaluate"` for full validation,
    and only unlock the test split after all choices are frozen.

Sections 5–8 in the notebook are older diagnostics that were already exercised
during development. `RUN_OPTIONAL_SETUP_DIAGNOSTICS=False` keeps them skipped,
including when using **Run all**. They are not required for the guided flow.

The verified local dataset statistics are:

| Split | Clips | Frames | Head boxes |
|---|---:|---:|---:|
| `train.txt` | 64 | 4,011 | 351,862 |
| `val.txt` | 15 | 724 | 64,888 |
| `test.txt` | 41 | 2,462 | 221,968 |

There are two zero-person frames in the training split; the adapter supports
them. No clip overlap was found between the three extended MDC++ splits.

## 4. Training

For the normal notebook workflow, select:

```python
ACTION = "train"
INITIALIZATION = "shanghaitech"  # or imagenet, scratch, custom
```

Read the preflight table and run the action cell. Training performs validation
automatically at `EVAL_FREQ`; no manual validation cell is needed between
epochs. With the notebook defaults, automatic early stopping is enabled only
after 30 epochs and stops after four consecutive periodic validation checks
without an MAE improvement. Since validation runs every five epochs, that is
20 epochs without improvement. This is a safety rule, not a claim that those
values are optimal.

The dashboard reads `training_history.csv` (with JSONL as a fallback) and shows
loss, periodic MAE/RMSE, epoch time, and peak GPU memory. Every row also records
learning rates, optimizer steps, source frames/patches, validation throughput,
the current best epoch, and the early-stopping counter. The best checkpoint's
epoch and metrics are written to `best_checkpoint.json`; the final status is
written to `training_summary.json`.

### Fine-tune from the included official P2PNet checkpoint

The notebook provides this command. Its equivalent is:

```bash
python train.py \
  --dataset_file MDC \
  --data_root /content/data/MovingDroneCrowd++ \
  --train_split train.txt \
  --val_split val.txt \
  --weights weights/SHTechA.pth \
  --output_dir "/content/drive/MyDrive/P2PNet_MDC/runs/p2pnet_mdc_run_001" \
  --epochs 100 \
  --lr_drop 80 \
  --batch_size 1 \
  --crop_size 512 \
  --num_patches 1 \
  --min_crop_points 1 \
  --eval_freq 5 \
  --val_frame_stride 5 \
  --save_every 10 \
  --early_stopping_patience 4 \
  --early_stopping_min_epochs 30 \
  --early_stopping_min_delta 0 \
  --num_workers 2 \
  --device cuda
```

`100` epochs is a practical starting schedule, not a claimed optimal MDC
hyperparameter. Do not copy the upstream `3500`-epoch ShanghaiTech schedule
without considering that MDC contains many more frames.

If CUDA runs out of memory, keep `batch_size=1` and reduce `crop_size` from 512
to 384 or 256. Do not solve training OOM by shrinking every full frame; that can
make already-small drone heads disappear.

`val_frame_stride=5` makes periodic validation affordable. Final reported
results must use `evaluate.py --frame_stride 1` on the complete validation/test
split.

### What the smoke and benchmark actions mean

- `smoke` is a correctness check, not a speed estimate or accuracy result. It
  deliberately uses one training batch and one validation frame.
- `benchmark` uses the real model/data path for a small fixed number of batches
  and validation frames. Each candidate runs in a fresh process so a failed or
  out-of-memory configuration does not contaminate the next one.
- The benchmark table estimates one full training epoch and one periodic
  validation pass from measured samples. Treat these as planning estimates;
  Colab load, caching, and longer runs can change them.
- A benchmark cannot determine whether `512 x 1` or `256 x 4` generalizes
  better. That requires comparable training runs and validation results.

### Train from ImageNet VGG initialization

Remove `--weights`. The default `--pretrained_backbone` then downloads the
standard torchvision ImageNet VGG16-BN weights. To train completely from random
initialization, also add `--no-pretrained_backbone`.

### Resume after a Colab reset

In the notebook, choose `ACTION="resume"` and leave
`RESUME_CHECKPOINT_CHOICE="latest"`. `EPOCHS` is the final total epoch number,
not the number of extra epochs.

Use the same run configuration and replace `--weights` with:

```bash
--resume "/content/drive/MyDrive/P2PNet_MDC/runs/p2pnet_mdc_run_001/checkpoints/latest.pth"
```

`latest.pth` contains model, optimizer, scheduler, epoch, best MAE/best epoch,
the early-stopping counter, and the original arguments. `best_mae.pth` is also
a complete checkpoint. `--weights` loads model parameters only; `--resume`
restores the complete training state, including early-stopping progress.

## 5. Validation and held-out testing

In the notebook, choose `ACTION="evaluate"`, `CHECKPOINT_CHOICE="best"`, and
`DATA_SPLIT="val"`. Evaluation writes a separate result directory whose name
records the split, checkpoint choice, threshold, and frame stride. Selecting
`DATA_SPLIT="test"` is blocked until `CONFIRM_FINAL_TEST=True`.

Validation:

```bash
python evaluate.py \
  --data_root /content/data/MovingDroneCrowd++ \
  --split_file val.txt \
  --weight_path /path/to/best_mae.pth \
  --output_dir /path/to/validation_results \
  --tile_size 1024 --tile_overlap 128 \
  --threshold 0.5 --frame_stride 1 --device cuda
```

Held-out test evaluation uses the identical command with
`--split_file test.txt`. Do not tune the confidence threshold on `test.txt`;
tune it on validation and freeze it before testing.

Outputs include:

- `metrics.json`: MAE, RMSE, command arguments, environment, and Git commit.
- `per_image.csv`: ground truth, prediction, and error for every frame.
- Optional marked images with `--visualize_first N`.

## 6. Single-image and directory inference

Single image:

```bash
python run_test.py \
  --input /path/to/image.jpg \
  --weight_path /path/to/checkpoint.pth \
  --output_dir /path/to/inference_output \
  --tile_size 1024 --tile_overlap 128 \
  --threshold 0.5 --device cuda
```

Directory inference:

```bash
python run_test.py \
  --input /path/to/images \
  --recursive \
  --weight_path /path/to/checkpoint.pth \
  --output_dir /path/to/batch_output \
  --tile_size 1024 --tile_overlap 128 \
  --threshold 0.5 --device cuda
```

Inference produces `predictions.csv`, one point/score JSON per image, marked
images, and `run.json` with reproducibility information.

For a quick functionality test, `--max_size 512 --tile_size 0` is fast. Do not
use that downscaled setting for final MDC measurements unless it is explicitly
part of the evaluated protocol.

## 7. Split selection

The default `train.txt`, `val.txt`, and `test.txt` files are the extended
MovingDroneCrowd++ protocol. To reproduce the older MovingDroneCrowd protocol,
select `MDC_train.txt`, `MDC_val.txt`, and `MDC_test.txt` explicitly. Never mix
the two protocols within one experiment.

## 8. Troubleshooting

- **Remote kernel cannot find local files:** push code to Git and let the
  notebook clone it; upload the dataset archive to Drive.
- **CUDA OOM during training:** lower crop size, then consider gradient
  accumulation as a future addition. Keep batch size at one.
- **CUDA OOM during evaluation:** lower `tile_size` to 768 or 512. Keep overlap
  smaller than tile size.
- **Very slow evaluation:** use `frame_stride=5` only for a development run,
  then use `1` for final results.
- **Wrong checkpoint architecture:** official and supplied checkpoints require
  the default `vgg16_bn`, `row=2`, and `line=2` settings.
- **Drive is slow:** verify that the dataset root is under `/content`, not the
  mounted Drive folder. Saving one checkpoint per epoch to Drive is acceptable;
  reading thousands of training images from Drive is not.
