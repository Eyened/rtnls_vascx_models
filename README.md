## VascX models

This repository contains the instructions for using the VascX models from the paper [VascX Models: Model Ensembles for Retinal Vascular Analysis from Color Fundus Images](https://arxiv.org/abs/2409.16016).

The model weights are in [huggingface](https://huggingface.co/Eyened/vascx).

<img src="imgs/CHASEDB1_12R_rgb.png" width="240" height="240" style="display:inline"><img src="imgs/CHASEDB1_12R.png" width="240" height="240" style="display:inline">

<img src="imgs/DRIVE_22_rgb.png" width="240" height="240" style="display:inline"><img src="imgs/DRIVE_22.png" width="240" height="240" style="display:inline">

<img src="imgs/HRF_04_g_rgb.png" width="240" height="240" style="display:inline"><img src="imgs/HRF_04_g.png" width="240" height="240" style="display:inline">

### Installation

To install the entire fundus analysis pipeline including fundus preprocessing, model inference code and vascular biomarker extraction:

1. Create a conda or virtualenv virtual environment, or otherwise ensure a clean environment.

2. Install the [rtnls_inference package](https://github.com/Eyened/retinalysis-inference).

### Usage

To speed up re-execution of vascx we recommend to run the preprocessing and segmentation steps separately:

1. Preprocessing. See [this notebook](./notebooks/0_preprocess.ipynb). This step is CPU-heavy and benefits from parallelization (see notebook).

2. Inference. See [this notebook](./notebooks/1_segment_preprocessed.ipynb). All models can be ran in a single GPU with >10GB VRAM.
