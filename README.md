## VascX models

This repo contains the Retinalysis models used in the VascX pipeline and instructions to extract 

### Installation

To install the entire fundus analysis pipeline including fundus preprocessing, model inference code and vascular biomarker extraction:

1. Create a conda or virtualenv virtual environment, or otherwise ensure a clean environment.

2. Install the [rtnls_inference package](https://github.com/Eyened/retinalysis-inference).

3. Download the vascx models and place them in the folder pointed by `$RTNLS_MODEL_RELEASES`.

### Usage

To speed up re-execution of vascx we recommend to run the preprocessing and segmentation steps separately:

1. Preprocessing. See [this notebook](./notebooks/0_preprocess.ipynb). This step is CPU-heavy and benefits from parallelization (see notebook).

2. Inference. See [this notebook](./notebooks/1_segment_preprocessed.ipynb). All models can be ran in a single GPU with >10GB VRAM.
