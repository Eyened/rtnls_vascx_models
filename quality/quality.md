---
library_name:
library_version:
---

## Model description
Quality estimation model ensemble (5 models) trained on the publicly available EyePACS EyeQ dataset.

## Intended uses
CFI quality control and quality estimation for AI and other automated analyses.

## Training data
We trained the model on the [EyePACS EyeQ dataset](https://github.com/hzfu/eyeq).

## Training procedure
See [VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images](https://tvst.arvojournals.org/article.aspx?articleid=2810436)

- Architecture: resnet50
- Weight initialisation: Imagenet
- Input resolution: 224px

## Evaluation
See [VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images](https://tvst.arvojournals.org/article.aspx?articleid=2810436)

## Limitations
- Trained on a single dataset - EyePACS.

## Inference

We recommend to run inference using [retinalysis-inference](https://github.com/Eyened/retinalysis-inference), which can auto-download the model weights and execute CFI pre-processing (crop to square) if necessary. For running inference on a CFI:

1. Install `retinalysis-inference`:
    ```
    pip install retinalysis-inference
    ```

2. Load the ensemble:
    ```
    from rtnls_inference import (
        HeatmapRegressionEnsemble,
        SegmentationEnsemble,
    )
    ensemble = SegmentationEnsemble.from_huggingface("Eyened/vascx:disc/av_july24.pt").to(device)
    ```

3. Run inference:
    ```
    # 1. Preprocess one input record.
    image = np.array(Image.open("/PATH/TO/RGB/FUNDUS/IMAGE.PNG"))
    item = ensemble.preprocess_item({"image": image})

    # 2. Make a batch of size 1.
    batch = ensemble.make_batch([item])

    # 3. Run the ensemble.
    predictions = ensemble.predict_batch(batch)

    # 4. Postprocess the prediction into a final mask.
    result = ensemble.postprocess_batch(predictions, batch)[0]
    ```

For more information see [retinalysis-inference](https://github.com/Eyened/retinalysis-inference).

## License & citation

If you use this model, please consider citing:

- [VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images](https://tvst.arvojournals.org/article.aspx?articleid=2810436)
Vargas Quiros, J. D., Liefers, B., van Garderen, K. A., Vermeulen, J. P., & Klaver, C. C. W. (2025). VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images. Translational Vision Science & Technology, 14(7), 19-19.

The original EyePACS datasets and EyeQ labels:
- [Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection) Emma Dugas, Jared, Jorge, and Will Cukierski. Diabetic Retinopathy Detection. 2015. Kaggle.
- [Evaluation of Retinal Image Quality Assessment Networks in Different Color-spaces](https://arxiv.org/abs/1907.05345)
Huazhu Fu, Boyang Wang, Jianbing Shen, Shanshan Cui, Yanwu Xu, Jiang Liu, Ling Shao. MICCAI, 2019


We share the model under a Creative Commons Attribution–NonCommercial 4.0 license. We note that some of the datasets used for training and evaluating the model ensemble were made publicly available for research purposes, but do not include a license. Other such as [Leuven-Haifa](https://www.nature.com/articles/s41597-024-03086-6) explicitly use custom terms with non-commercial clauses. 
