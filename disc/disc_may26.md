---
library_name: retinalysis-inference
library_version: 
---

## Model description

UNet ensemble (5 models) for optic disc segmentation from color fundus images (CFIs), trained on a variety of public datasets, and on images from the Rotterdam Study. 

## Intended uses

Robust segmentation of the optic disc boundaries on color fundus images. Useful for localisation and delineation of the optic disc.

## Training data

We used the publicly available [ORIGA](https://pubmed.ncbi.nlm.nih.gov/21095735/), [PAPILA](https://www.nature.com/articles/s41597-022-01388-1), [IDRID](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid), [ADAM](https://ieeexplore.ieee.org/document/9768802), [PALM](https://www.nature.com/articles/s41597-024-02911-2), [REFUGE2](https://ieee-dataport.org/documents/refuge2-challenge-treasure-trove-multi-dimension-analysis-and-evaluation-glaucoma), together with internal Rotterdam Study images. Peripapillary atrophy (PPA) is in general not considered part of the disc and excluded from the segmentations. If this model helps you, please consider citing the authors of these datasets.

## Training procedure

See [VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images](https://tvst.arvojournals.org/article.aspx?articleid=2810436)

## Evaluation

See [VascX Models: Deep Ensembles for Retinal Vascular Analysis From Color Fundus Images](https://tvst.arvojournals.org/article.aspx?articleid=2810436)

Notes:
- The model was trained and evaluated using CFIs cropped to a square using [retinalysis-fundusprep].

## Limitations

- Does not segment the optic cup
- Not trained on WF or UWF fundus images
- May not generalize well to images with different characteristics

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
    ensemble = SegmentationEnsemble.from_huggingface("Eyened/vascx:disc/disc_july24.pt").to(device)
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