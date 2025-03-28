---
license: agpl-3.0
pipeline_tag: image-segmentation
tags:
- medical
- biology
---

# 👁️ VascX models

This repository contains the instructions for using the VascX models from the paper [VascX Models: Model Ensembles for Retinal Vascular Analysis from Color Fundus Images](https://arxiv.org/abs/2409.16016).

The model weights are in [huggingface](https://huggingface.co/Eyened/vascx) but will be downloaded automatically (see instructions below).

<img src="imgs/samples_vascx_hrf.png">

## 🛠️ Installation

To install the entire fundus analysis pipeline including fundus preprocessing, model inference code and vascular biomarker extraction:

1. Create a conda or virtualenv virtual environment, or otherwise ensure a clean environment.

2. Make sure that cuda and torchvision are available, or install them using your preferred method. For example:

   ```
   pip3 install torch torchvision torchaudio  # pip and CUDA 12
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia # conda and CUDA 12
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # pip and CUDA 11
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia # conda and CUDA 11
   ```

2. Install the [retinalysis-inference package](https://github.com/Eyened/retinalysis-inference).

   ```
   pip install retinalysis-fundusprep
   pip install retinalysis-inference
   ```

3. Clone and install this repository:

```
git clone https://github.com/Eyened/rtnls_vascx_models.git -b github
cd rtnls_vascx_models
pip install -e .
```

## 🚀 `vascx run` Command

The `run` command is the most simple way to run the VascX models on locally-accessible images by providing a folder path or a list of files.

### Usage

```bash
vascx run DATA_PATH OUTPUT_PATH [OPTIONS]
```

### Arguments

- `DATA_PATH`: Path to input data. Can be either:
  - A directory containing fundus images
  - A CSV file with a 'path' column containing paths to images

- `OUTPUT_PATH`: Directory where processed results will be stored

### Options

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--preprocess/--no-preprocess` | `--preprocess` | Run preprocessing to standardize images for model input |
| `--vessels/--no-vessels` | `--vessels` | Run vessel segmentation and artery-vein classification |
| `--disc/--no-disc` | `--disc` | Run optic disc segmentation |
| `--quality/--no-quality` | `--quality` | Run image quality assessment |
| `--fovea/--no-fovea` | `--fovea` | Run fovea detection |
| `--overlay/--no-overlay` | `--overlay` | Create visualization overlays combining all results |
| `--n_jobs` | `4` | Number of preprocessing workers for parallel processing |
| `--devices` | None | Comma separated list of GPU ids to use (e.g., '0,1,2') |
| `--av_model` | `"hf@Eyened/vascx:artery_vein/av_july24.pt"` | Model to use for artery-vein segmentation |
| `--vessels_model` | `"hf@Eyened/vascx:vessels/vessels_july24.pt"` | Model to use for vessel segmentation |
| `--disc_model` | `"hf@Eyened/vascx:disc/disc_july24.pt"` | Model to use for disc segmentation |
| `--quality_model` | `"hf@Eyened/vascx:quality/quality.pt"` | Model to use for quality estimation |
| `--fovea_model` | `"hf@Eyened/vascx:fovea/fovea_july24.pt"` | Model to use for fovea detection |

### 🧠 Model Weights

By default, model weights are automatically downloaded from the [Eyened/vascx](https://huggingface.co/Eyened/vascx/tree/main) Hugging Face repository when you first run the command. In addition to the default weights, we made available model weights with one dataset left out, meant for benchmarking and reproduction of the results in our paper. For example see [artery_vein weights](https://huggingface.co/Eyened/vascx/tree/main/artery_vein), where `av_july24_RS.pt` are the artery-vein segmentation weights trained without the Rotterdam Study set. To run using these weights specify `--av_model hf@Eyened/vascx:artery_vein/av_july24_RS.pt`.

You can also use local model weights by specifying an absolute path model files. For detailed instructions refer to the [retinalysis-inference repository](https://github.com/Eyened/retinalysis-inference).

### 🖥️ GPU Utilization

Use the `--devices` option to specify which GPUs to use for inference:

```bash
# Use GPU 0
vascx run /path/to/images /path/to/output --devices 0

# Use multiple GPUs
vascx run /path/to/images /path/to/output --devices 0,1,2
```

### 📁 Output Structure

When run with default options, the command creates the following structure in `OUTPUT_PATH`:

```
OUTPUT_PATH/
├── preprocessed_rgb/     # Standardized fundus images
├── vessels/              # Vessel segmentation results
├── artery_vein/          # Artery-vein classification
├── disc/                 # Optic disc segmentation
├── overlays/             # Visualization images
├── bounds.csv            # Image boundary information
├── quality.csv           # Image quality scores
└── fovea.csv             # Fovea coordinates
```

### 🔄 Processing Stages

1. **Preprocessing**: 
   - Standardizes input images for consistent analysis
   - Outputs preprocessed images and boundary information

2. **Quality Assessment**:
   - Evaluates image quality with three quality metrics (q1, q2, q3)
   - Higher scores indicate better image quality

3. **Vessel Segmentation and Artery-Vein Classification**:
   - Identifies blood vessels in the retina
   - Classifies vessels as arteries (1) or veins (2) with intersections (3)

4. **Optic Disc Segmentation**:
   - Identifies the optic disc location and boundaries

5. **Fovea Detection**:
   - Determines the coordinates of the fovea (center of vision)

6. **Visualization Overlays**:
   - Creates color-coded images showing:
     - Arteries in red
     - Veins in blue
     - Optic disc in white
     - Fovea marked with yellow X

### 💻 Examples

**Process a directory of images with all analyses:**
```bash
vascx run /path/to/images /path/to/output
```

**Process specific images listed in a CSV:**
```bash
vascx run /path/to/image_list.csv /path/to/output
```

**Only run preprocessing and vessel segmentation:**
```bash
vascx run /path/to/images /path/to/output --no-disc --no-quality --no-fovea --no-overlay
```

**Skip preprocessing on already preprocessed images:**
```bash
vascx run /path/to/preprocessed/images /path/to/output --no-preprocess
```

**Increase parallel processing workers:**
```bash
vascx run /path/to/images /path/to/output --n_jobs 8
```

### 📝 Notes

- The CSV input must contain a 'path' column with image file paths
- If the CSV includes an 'id' column, these IDs will be used instead of filenames
- When `--no-preprocess` is used, input images must already be in the proper format
- The overlay visualization requires at least one analysis component to be enabled

## 📓 Notebooks <a id="notebooks"></a>

For more advanced usage, we have Jupyter notebooks showing how preprocessing and inference are run.

To speed up re-execution of vascx we recommend to run the preprocessing and segmentation steps separately:

1. Preprocessing. See [this notebook](./notebooks/0_preprocess.ipynb). This step is CPU-heavy and benefits from parallelization (see notebook).

2. Inference. See [this notebook](./notebooks/1_segment_preprocessed.ipynb). All models can be ran in a single GPU with >10GB VRAM.
