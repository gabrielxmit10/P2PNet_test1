# P2PNet–MDC changes

The original network definition is preserved. The changes are dataset,
orchestration, compatibility, evaluation, and checkpoint layers around it,
plus one correction that activates the intended point-regression loss.

## Major additions

### Native MovingDroneCrowd++ adapter

Files: `crowd_datasets/MDC/mdc.py`, `crowd_datasets/MDC/__init__.py`, and
`crowd_datasets/__init__.py`.

- Reads the original `frames/`, `annotations/`, and split files directly.
- Expands both scene-level (`scene_2`) and clip-level (`scene_4/1`) split entries.
- Maps image `N.jpg` to annotation `frame_id=N-1`.
- Converts head boxes to centre points in memory.
- Supports zero-person frames without modifying the source dataset.
- Uses memory-efficient random scaled crops: it crops before resizing instead
  of resizing an entire 4K image for every training sample.
- Makes patch size, number of patches, frame stride, and positive-patch sampling
  configurable.

### Evaluation and general inference

Files: `evaluate.py`, `run_test.py`, and `workflow_utils.py`.

- Replaced the one-hardcoded-image CUDA demo with file and directory inference.
- Added CPU/automatic/CUDA device selection.
- Added checkpoint-format normalization for `model`, `state_dict`, raw state
  dictionaries, and repeated `module.` prefixes.
- Added overlap-aware tiled inference for 720p–4K frames.
- Saves machine-readable point predictions, per-image counts, visualizations,
  frame-level MAE/RMSE, and reproducibility metadata.

### Resumable training and inexpensive smoke tests

Files: `train.py`, `smoke_test.py`, and `validate_mdc.py`.

- Training checkpoints now include optimizer, scheduler, epoch, best metric,
  model parameters, and arguments.
- Added separate `--weights` (model-only initialization) and `--resume`
  (complete recovery) semantics.
- Added bounded sample/batch options for inexpensive tests.
- Added configurable periodic tiled validation.
- Added optional validation-MAE early stopping with a minimum-epoch guard,
  configurable patience/minimum improvement, and resumable patience state.
- Periodic validation now switches to evaluation mode without updating VGG
  batch-normalization statistics, then restores training mode.
- Training-time validation now correctly obtains the validation split and frame
  stride from the training arguments.
- Added dataset layout, split-overlap, annotation, frame-mapping, and image
  readability validation.
- Records environment, command, Git revision, and complete configuration.
- Writes `best_checkpoint.json` with the selected epoch, MAE/RMSE, threshold,
  split, and frame stride alongside `best_mae.pth`.
- Writes a human-readable `training_history.csv` plus JSONL with per-epoch
  losses, learning rates, optimizer steps, time, validation throughput, and
  peak CUDA-memory measurements. `training_summary.json` records why and when
  a run finished.

### Colab orchestration

Files: `colab_runner.ipynb`, `requirements-colab.txt`, and
`P2PNET_MDC_GUIDE.md`.

- Stages code into `/content` from Git.
- Mounts Drive, copies/extracts the dataset into `/content`, and checks disk.
- Keeps checkpoints, TensorBoard data, metrics, and predictions in Drive.
- Replaces several independent Boolean switches with one `ACTION` control:
  inspect, smoke, benchmark, train, resume, evaluate, or inference.
- Resolves initialization/checkpoint/split choices in a preflight table before
  executing anything and guards the held-out test split behind confirmation.
- Adds a training dashboard, best-checkpoint metadata display, worst-frame
  evaluation table, and saved inference/evaluation visualization viewer.
- The new smoke action exercises a 512-pixel batch and training-time periodic
  validation rather than bypassing the validation path, and reports time and
  memory diagnostics.
- Adds a guided technical benchmark comparing `512 x 512, 1 patch` with
  `256 x 256, 4 patches` in isolated short runs. It reports measured resource
  use and rough full-run timing estimates without presenting them as accuracy
  evidence.
- Gates the older diagnostic cells behind one disabled-by-default switch, so
  `ACTION="inspect"` remains non-training even when the notebook uses **Run all**.

## Important compatibility and correctness fixes

### ImageNet VGG loading

File: `models/backbone.py`.

The upstream custom VGG helper pointed to private absolute paths under
`/apdcephfs/private/...`. Backbone initialization now uses torchvision's public
weights API. Loading a complete P2PNet checkpoint disables the unnecessary
ImageNet download.

### Point loss was excluded upstream

File: `models/p2pnet.py`.

The criterion returned `loss_point`, while the weight dictionary used the key
`loss_points`. The training loop only optimizes losses present in the weight
dictionary, so the localization loss was silently omitted. The key is now
consistently `loss_point`. This is the only intentional correction to training
semantics.

### Device-safe anchor points

File: `models/p2pnet.py`.

Anchor tensors now follow the input tensor's device instead of moving to CUDA
whenever any CUDA device exists. This enables reliable explicit CPU tests.

### Modern torchvision import

File: `util/misc.py`.

The upstream code parsed only the first three characters of the torchvision
version. For example, `0.29` became `0.2`, causing imports of removed private
operators. The obsolete branch was replaced with modern PyTorch interpolation.
The missing `dim` argument in a softmax call was also made explicit.

## Smaller changes

- Created output/checkpoint directories automatically.
- Removed hardcoded CUDA-only execution from workflow entry points.
- Made the confidence threshold configurable.
- Added modern Pillow resampling in place of removed `Image.ANTIALIAS`.
- Added `.gitignore` entries for generated results, caches, and dataset archives.
- Added separate train and periodic-validation frame strides.
- Preserved the original SHHA loader and model files where no change was needed.

## Validation performed during implementation

On Windows, Python 3.11, PyTorch 2.14 CPU, torchvision 0.29 CPU:

- Full split/layout validation succeeded for all 7,197 frames.
- Official `weights/SHTechA.pth` loaded strictly (123 state tensors).
- MDC sample loading and a model forward/backward pass succeeded with active
  classification and point losses.
- Single-image inference completed and saved CSV, JSON, and visualization.
- Two-frame MDC evaluation completed and saved metrics/per-frame results.
- A one-batch training run saved a complete checkpoint.
- A second run resumed optimizer/scheduler/epoch state from that checkpoint.
- A bounded two-epoch run exercised history CSV creation, timing/memory fields,
  automatic early stopping, summary creation, and persistence of the stopping
  state in `latest.pth`.

No claim is made that CPU smoke-test predictions are accurate on MDC. Accuracy
requires MDC training and full GPU evaluation.
