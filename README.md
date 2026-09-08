DATASET USED - 375-V1 - https://www.kaggle.com/datasets/d07028a5f290c1ed2f58b48beaaaf50fa8577e409d4d1b7bb27fe40bece9d2cf
KAGGLE NOTEBOOK - https://www.kaggle.com/code/prachi232005/notebook0b309e84c3

# Turf Detection — UNet + MobileSAM Combined (375v1 dataset) — Local Setup

One notebook, three parts, run top to bottom against the `375v1` dataset
(already split into `train/images`, `train/masks`, `val/images`, `val/masks`):

1. Train UNet (positive-focused: weighted loss + oversampling turf tiles)
2. Fine-tune MobileSAM's mask decoder for clean boundaries
3. Combine: UNet finds *where* turf is → MobileSAM refines *how it's shaped*
   → polygon smoothing for the final output

## Setup

1. **Python 3.10/3.11**, then a venv:
```bash
python3 -m venv turf_env
source turf_env/bin/activate      # mac/linux
turf_env\Scripts\activate         # windows
```

2. **Install PyTorch first** (pick your system's command from
https://pytorch.org/get-started/locally/):
```bash
# NVIDIA GPU example
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch torchvision
```

3. **Install the rest** (the notebook's own install cell only covers some of
these — install everything up front to avoid mid-run surprises):
```bash
pip install segmentation-models-pytorch albumentations opencv-python-headless \
            scipy matplotlib jupyter numpy
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

4. **Get the `375v1` dataset on disk**, keeping its folder structure:
```
375v1/
  train/
    images/
    masks/
  val/
    images/
    masks/
```

5. **Get a MobileSAM checkpoint** (`mobile_sam.pt`) — download from the
MobileSAM GitHub repo's releases if you don't already have it. The notebook
expects it at `SAM_CHECKPOINT_BASE` (see config below) — place it there, or
add a download step like:
```bash
wget -O ./turf_project_combined/mobile_sam.pt \
  https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```
(create the `turf_project_combined` folder first, or point `BASE_DIR` at
wherever you put it).

## Run

1. `jupyter notebook`, open the notebook
2. Edit the config cell (cell 1):
   ```python
   DATASET_ROOT = "/path/to/375v1"
   ```
   Everything else (`TRAIN_IMG_DIR`, `VAL_IMG_DIR`, etc.) derives from this
   automatically — no other path edits needed if your folder structure
   matches the layout above.
3. **Delete or skip the two debug cells (3 and 4)** — these were added while
   troubleshooting a Kaggle path issue (`print("DATASET_ROOT:", ...)` and the
   `os.walk("/kaggle/input")` tree dump). They're not part of the actual
   pipeline and cell 4 is Kaggle-specific (walks `/kaggle/input`, which
   won't exist locally — harmless if run, it'll just print nothing found,
   but you don't need it).
4. Run the rest top to bottom:
   - **Part 1** trains UNet — checks mask polarity/positive-pixel ratio
     first, then trains with weighted BCE + Tversky loss and oversampling of
     turf-containing tiles, saving the best checkpoint to
     `UNET_SAVE_PATH`
   - **Part 2** fine-tunes MobileSAM's mask decoder (image encoder + prompt
     encoder frozen) using box prompts derived from ground-truth masks,
     saving to `SAM_DECODER_SAVE_PATH`
   - **Part 3** chains them: UNet's coarse prediction → box prompt per
     detected blob → MobileSAM refines the boundary → polygon smoothing →
     final overlay results + mean IoU, saved to `RESULTS_DIR`

## Roughly how long it takes

This trains two models in sequence in one run, so total time is the sum of
both:

- **UNet training (GPU)**: ~1.5-3 hours (frozen warmup + up to 25 epochs
  unfrozen, with early stopping — often finishes sooner)
- **MobileSAM fine-tuning (GPU)**: comparable or longer per epoch than UNet
  since it processes one image at a time internally — `SAM_EPOCHS` defaults
  to 10; lower it (e.g. to 5) if you're short on time, since the combined
  pipeline's quality leans more on UNet's detection than on SAM being
  perfectly fine-tuned
- **CPU only**: not realistic for this notebook — fine-tuning SAM-family
  models on CPU is impractically slow (the notebook prints a warning if it
  detects no GPU). Use Kaggle/Colab GPU if you don't have a local one.

## Notes

- Mask polarity check (Part 1, right after Part 1 markdown) will show you a
  few sample tiles — confirm the white regions in the masks actually
  correspond to turf/lawn in the image before trusting the rest of the run
- Tune `pos_weight_value` (computed automatically from your masks) and the
  Tversky loss `alpha`/`beta` values if the model still under- or
  over-predicts turf after training
- Polygon smoothing (Part 3) is cosmetic only — cleans up jagged mask edges
  into smooth curves, doesn't affect IoU. Tune `min_area` and
  `spline_smoothing` inside `smooth_mask()` if needed
- Mac M1/M2/M3: falls back to CPU unless you manually switch to `mps` in the
  config cell:
  ```python
  device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
  ```
  set `use_amp = False` if `autocast` errors on `mps`

## Common issues

- **`FileNotFoundError` on `DATASET_ROOT`** → double-check the path and
  folder structure match what's listed above
- **MobileSAM import/install fails** → make sure you ran the
  `git+https://github.com/ChaoningZhang/MobileSAM.git` install and have
  `mobile_sam.pt` downloaded to the path in `SAM_CHECKPOINT_BASE`
- **CUDA out of memory** → drop `batch_size` in the UNet DataLoader (16→8),
  or lower `SAM_BATCH_SIZE` (4→2) in Part 2
- **Part 3 errors on missing checkpoints** → Parts 1 and 2 must fully
  complete first; Part 3 loads both saved checkpoints from disk
- **UNet detects little/no turf** → check the mask polarity visual output
  from Part 1 early on; an inverted mask convention would cause exactly this
