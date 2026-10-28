# Data

No data is committed to this repository. `data/raw/` and `data/processed/` are git-ignored.

## Real Tactile MNIST (default)

| | |
|---|---|
| Dataset | `TimSchneider42/tactile-mnist-touch-real-single-t256-64x64` on Hugging Face |
| Sensor | GelSight Mini (markerless), mounted on a Franka Research 3 arm |
| Content | 600 3D-printed MNIST digits x 256 touches = 153,600 RGB frames (64x64, downscaled from 320x240) |
| Labels | `label`: digit class 0-9 of the touched object; `object_id`: physical object |
| Official splits | `train`: 500 objects, `test`: 100 objects (no object overlap; 10 objects per class in test) |
| License | CC-BY-2.0 |
| Citation | T. Schneider et al., *Tactile MNIST: Benchmarking Active Tactile Perception*, arXiv:2506.06361, 2025 |

Each touch was made at a uniformly random position inside a 12x12 cm cell containing the digit, pressing until the
normal force exceeded 5 N. **Many touches therefore hit the rubber mat next to the digit and show an undeformed
gel.** Such frames carry no information about the class; see the main README for how they are handled.

### Download and prepare

```bash
python scripts/prepare_data.py
```

This downloads the two parquet files (~133 MB) to `data/raw/tactile-mnist-real-64/`, decodes all frames, checks for
undecodable and duplicate images, verifies that every object has a single label, reports the class balance and
writes a compressed cache to `data/processed/tactile_mnist_real_64.npz`.

Manual alternative: download `data/train-00000-of-00001.parquet` and `data/test-00000-of-00001.parquet` from
<https://huggingface.co/datasets/TimSchneider42/tactile-mnist-touch-real-single-t256-64x64> into
`data/raw/tactile-mnist-real-64/` and run `python scripts/prepare_data.py --skip-download`.

### Fields used

| Parquet column | Use |
|---|---|
| `sensor_image` (list of PNG bytes, one per touch) | model input |
| `label` | target class |
| `object_id` | grouping key: all touches of one object stay in one split |
| `gel_pose_cell_frame.position` (z component) | **analysis only** - used to sanity-check the image-based contact detector against the robot's measured gel height; never a model input |

## Using your own sensor data (folder-per-class)

Set `data.source: folder` and `data.folder_root` in `config.yaml` and arrange images as

```
data/folder_dataset/
  <class_a>/
    <recording-id>__<frame>.png
  <class_b>/
    ...
```

The part of the file name before `__` identifies the recording or physical trial. All frames sharing it are kept in
the same split to prevent leakage between near-identical consecutive frames. Without `__`, every file is treated as
an independent sample (and a warning-free but leakage-prone split may result if frames are correlated).
The no-contact reference frame is estimated from the training images, so frames from different sensors should not be
mixed in one dataset.
