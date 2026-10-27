---
license: agpl-3.0
pipeline_tag: image-segmentation
tags:
- medical
- biology
- ophthalmology
- retina
---

> [!IMPORTANT]
> The entire VascX pipeline, including feature extraction is available here: [retinalysis-vascx](https://github.com/eyened/retinalysis-vascx). A new preprint presenting the pipeline is [available here](https://arxiv.org/abs/2602.08580). We will stop supporting this repository and move to the new one linked above. We will keep updating the models and pipeline there.

# 👁️ VascX models

This repository contains model weights and model cards for the VascX models from the paper [VascX Models: Model Ensembles for Retinal Vascular Analysis from Color Fundus Images](https://arxiv.org/abs/2409.16016).

The legacy `vascx_models` Python package and CLI have been removed. For inference and the complete analysis pipeline, use [retinalysis-vascx](https://github.com/eyened/retinalysis-vascx).

The model weights are in [huggingface](https://huggingface.co/Eyened/vascx).

<img src="imgs/samples_vascx_hrf.png">

## Model cards

Please see the model card for each model for details about its training procedure, datasets and ciations. For example:
- [Artery-vein](artery_vein/av_july24.md)
- [Optic disc](disc/disc_july24.md)
- [Fovea detection](fovea/fovea_july24.md)
- [Quality estimation](quality/quality.md)

## Licensing

We share the models under a Creative Commons Attribution–NonCommercial 4.0 license. We note that some of the datasets used for training and evaluating the model ensembles were made publicly available for research purposes only. We encourage due diligence for any applications with potential for commercial use.