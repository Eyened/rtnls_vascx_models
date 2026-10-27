---
library_name:
library_version:
---

## Model description
UNet ensemble (5 models) for fovea localisation from color fundus images (CFIs), trained on various public datasets, and on images from the Rotterdam Study. The model uses heatmap regression to produce (x,y) coordinates for the fovea location in the image, with x and y in the range [0,1], relative to image size.

## Intended uses
Localisation of the fovea on color fundus images.

## Training data
The model was trained on [IDRID](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), [ADAM](https://ieeexplore.ieee.org/document/9768802), [PALM](https://www.nature.com/articles/s41597-024-02911-2), [REFUGE2](https://ieee-dataport.org/documents/refuge2-challenge-treasure-trove-multi-dimension-analysis-and-evaluation-glaucoma), and on images from the Rotterdam Study.

## Training procedure
See [VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images](https://tvst.arvojournals.org/article.aspx?articleid=2810436)

## Evaluation
See [VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images](https://tvst.arvojournals.org/article.aspx?articleid=2810436)

Notes:
- The model was trained and evaluated using CFIs cropped to a square using [retinalysis-fundusprep].

## Limitations
- The fovea may be difficult to localise precisely on some CFIs, depending on anatomy and illumination.
- Does not work when the fovea is out of bounds.

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
    ensemble = SegmentationEnsemble.from_huggingface("Eyened/vascx:disc/fovea_july24.pt").to(device)
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

- [retinalysis-vascx: An explainable software toolbox for the extraction of retinal vascular biomarkers](https://arxiv.org/abs/2602.08580)
Vargas Quiros, J. V., Beyeler, M. J., Vela, S. O., Center, E. R., Bergmann, S., Klaver, C. C. W., & Liefers, B. (2026). retinalysis-vascx: An explainable software toolbox for the extraction of retinal vascular biomarkers. arXiv preprint arXiv:2602.08580.

We share the model under a Creative Commons Attribution–NonCommercial 4.0 license. We note that some of the datasets used for training and evaluating the model ensemble were made publicly available for research purposes, but do not include a license.