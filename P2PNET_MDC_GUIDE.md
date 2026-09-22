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
  -C "C:\Users\GabrielSchmitz\Downloads\UFRJ\00_IC\一00_Before\05_VisDrone_Dataset\MovingDroneCrowd" `
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
        └── history.jsonl
```

The official `weights/SHTechA.pth` checkpoint is already in this repository.
It is useful for checking inference and as an MDC fine-tuning initialization,
but its ShanghaiTech predictions are not valid MDC benchmark results until the
model has been trained/fine-tuned on MDC.

## 2. Open and configure the notebook

Open `colab_runner.ipynb` in VS Code, select the Google Colab GPU kernel, and
change only the central configuration cell first:

```python
REPO_URL = "https://github.com/YOUR_ACCOUNT/YOUR_P2PNET_FORK.git"
REPO_REF = "your-mdc-branch"
DATASET_ARCHIVE = "/content/drive/MyDrive/P2PNet_MDC/data/MovingDroneCrowd++.tar"
RUN_NAME = "p2pnet_mdc_run_001"
```

Run the setup/staging cells in order. After a runtime reset, repeat setup and
dataset staging. Checkpoints in Drive survive the reset.

## 3. Required validation order

Run these notebook sections before full training:

1. Runtime/GPU report.
2. Repository checkout and dependency installation.
3. Dataset staging and `validate_mdc.py`.
4. One annotated sample visualization.
5. Model/dataset forward-backward `smoke_test.py`.
6. Official-checkpoint single-image inference.
7. Two-frame validation evaluation.
8. One-batch, one-epoch training smoke run.
9. Full training only after all previous cells succeed.

The verified local dataset statistics are:

| Split | Clips | Frames | Head boxes |
|---|---:|---:|---:|
| `train.txt` | 64 | 4,011 | 351,862 |
| `val.txt` | 15 | 724 | 64,888 |
| `test.txt` | 41 | 2,462 | 221,968 |

There are two zero-person frames in the training split; the adapter supports
them. No clip overlap was found between the three extended MDC++ splits.

## 4. Training

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

### Train from ImageNet VGG initialization

Remove `--weights`. The default `--pretrained_backbone` then downloads the
standard torchvision ImageNet VGG16-BN weights. To train completely from random
initialization, also add `--no-pretrained_backbone`.

### Resume after a Colab reset

Use the same run configuration and replace `--weights` with:

```bash
--resume "/content/drive/MyDrive/P2PNet_MDC/runs/p2pnet_mdc_run_001/checkpoints/latest.pth"
```

`latest.pth` contains model, optimizer, scheduler, epoch, best MAE, and the
original arguments. `best_mae.pth` is also a complete checkpoint. `--weights`
loads model parameters only; `--resume` restores the complete training state.

## 5. Validation and held-out testing

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

