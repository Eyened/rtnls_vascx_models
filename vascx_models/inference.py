import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from rtnls_inference.dtos_inference import ModelInputDTO
from rtnls_inference.ensembles.ensemble_classification import ClassificationEnsemble
from rtnls_inference.ensembles.ensemble_heatmap_regression import (
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from tqdm import tqdm


def _create_dtos(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
) -> List[ModelInputDTO]:
    """Helper to create ModelInputDTOs from paths."""
    if ids is None:
        ids = [p.stem for p in rgb_paths]

    if ce_paths is None:
        ce_paths = [None] * len(rgb_paths)

    if len(rgb_paths) != len(ce_paths):
        raise ValueError("rgb_paths and ce_paths must have the same length")

    if len(rgb_paths) != len(ids):
        raise ValueError("rgb_paths and ids must have the same length")

    return [
        ModelInputDTO(
            id=str(id_val),
            image=str(rgb_path),
            contrast_enhanced=str(ce_path) if ce_path else None,
        )
        for id_val, rgb_path, ce_path in zip(ids, rgb_paths, ce_paths)
    ]


def iterate_quality_estimation(
    data: List[ModelInputDTO],
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> Iterator[Dict[str, Any]]:
    """Yield quality ensemble inference items."""
    ensemble_quality = ClassificationEnsemble.from_huggingface(
        "Eyened/vascx:quality/quality.pt"
    ).to(device)

    inputs = {"images": [item.to_serialized_dict() for item in data]}
    dataloader = ensemble_quality._make_inference_dataloader(
        inputs,
        num_workers=8,
        preprocess=False,
        batch_size=16,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            items = ensemble_quality._predict_batch(batch)

            for item in items:
                yield item


def run_quality_estimation(
    fpaths,
    ids: Optional[List[str]] = None,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    data = _create_dtos(fpaths, ids=ids)
    output_ids, outputs = [], []

    for item in iterate_quality_estimation(data, device=device):
        output_ids.append(item["id"])
        outputs.append(item["logits"].tolist())

    return pd.DataFrame(
        outputs,
        index=output_ids,
        columns=["q1", "q2", "q3"],
    )


def iterate_segmentation_vessels_and_av(
    data: List[ModelInputDTO],
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    predict_av: bool = True,
    predict_vessels: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Yield raw segmentation items for AV and vessels."""
    if not predict_av and not predict_vessels:
        return

    ensemble_av = (
        # SegmentationEnsemble.from_huggingface("Eyened/vascx:artery_vein/av_july24.pt")
        SegmentationEnsemble.from_huggingface("Eyened/vascx:artery_vein/av_july24_2.pt")
        .to(device)
        .eval()
        if predict_av
        else None
    )
    ensemble_vessels = (
        SegmentationEnsemble.from_huggingface("Eyened/vascx:vessels/vessels_july24.pt")
        .to(device)
        .eval()
        if predict_vessels
        else None
    )
    reference_ensemble = ensemble_av or ensemble_vessels
    if reference_ensemble is None:
        return

    inputs = {"images": [item.to_serialized_dict() for item in data]}
    dataloader = reference_ensemble._make_inference_dataloader(
        inputs,
        num_workers=16,
        preprocess=False,
        batch_size=16,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            items_av = ensemble_av._predict_batch(batch) if ensemble_av else None
            items_vessels = (
                ensemble_vessels._predict_batch(batch) if ensemble_vessels else None
            )

            num_items = (
                len(items_av)
                if items_av is not None
                else len(items_vessels)
                if items_vessels is not None
                else 0
            )

            for idx in range(num_items):
                av_item = items_av[idx] if items_av is not None else None
                vessel_item = items_vessels[idx] if items_vessels is not None else None

                yield {
                    "id": (
                        av_item["id"]
                        if av_item is not None
                        else vessel_item["id"]
                        if vessel_item is not None
                        else None
                    ),
                    "av": av_item,
                    "vessels": vessel_item,
                }


def run_segmentation_vessels_and_av(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    av_path: Optional[Path] = None,
    vessels_path: Optional[Path] = None,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    predict_av: bool = False,
    predict_vessels: bool = False,
) -> None:
    """
    Run AV and vessel segmentation on the provided images.

    Args:
        rgb_paths: List of paths to RGB fundus images
        ce_paths: Optional list of paths to contrast enhanced images
        ids: Optional list of ids to pass to _make_inference_dataloader
        av_path: Folder where to store output AV segmentations
        vessels_path: Folder where to store output vessel segmentations
        device: Device to run inference on
        callback: Optional callback to process results instead of saving them
        predict_av: Whether to predict AV segmentation (default False, overriden by av_path)
        predict_vessels: Whether to predict vessel segmentation (default False, overriden by vessels_path)
    """
    if av_path is not None:
        av_path.mkdir(exist_ok=True, parents=True)
    if vessels_path is not None:
        vessels_path.mkdir(exist_ok=True, parents=True)

    should_predict_av = (av_path is not None) or predict_av
    should_predict_vessels = (vessels_path is not None) or predict_vessels

    data = _create_dtos(rgb_paths, ce_paths=ce_paths, ids=ids)

    for result in iterate_segmentation_vessels_and_av(
        data,
        device=device,
        predict_av=should_predict_av,
        predict_vessels=should_predict_vessels,
    ):
        if callback is not None:
            callback(result)
        else:
            if av_path is not None and result["av"] is not None:
                fpath = os.path.join(av_path, f"{result['id']}.png")
                mask = np.argmax(result["av"]["image"], -1)
                Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)

            if vessels_path is not None and result["vessels"] is not None:
                fpath = os.path.join(vessels_path, f"{result['id']}.png")
                mask = np.argmax(result["vessels"]["image"], -1)
                Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)


def iterate_segmentation_disc(
    data: List[ModelInputDTO],
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> Iterator[Dict[str, Any]]:
    """Yield disc segmentation inference items."""
    ensemble_disc = (
        SegmentationEnsemble.from_huggingface("Eyened/vascx:disc/disc_july24.pt")
        .to(device)
        .eval()
    )

    inputs = {"images": [item.to_serialized_dict() for item in data]}
    dataloader = ensemble_disc._make_inference_dataloader(
        inputs,
        num_workers=16,
        preprocess=False,
        batch_size=32,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            items = ensemble_disc._predict_batch(batch)
            items = [dataloader.dataset.transform.undo_item(item) for item in items]

            for item in items:
                yield item


def run_segmentation_disc(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> None:
    if output_path is None and callback is None:
        raise ValueError(
            "Either output_path or callback must be provided for disc segmentation"
        )

    if output_path is not None:
        output_path.mkdir(exist_ok=True, parents=True)

    data = _create_dtos(rgb_paths, ce_paths=ce_paths, ids=ids)

    for item in iterate_segmentation_disc(data, device=device):
        if callback is not None:
            callback(item)
        elif output_path is not None:
            fpath = os.path.join(output_path, f"{item['id']}.png")
            mask = np.argmax(item["image"], -1)
            Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)


def iterate_fovea_detection(
    data: List[ModelInputDTO],
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> Iterator[Dict[str, Any]]:
    """Yield fovea detection inference items."""
    ensemble_fovea = HeatmapRegressionEnsemble.from_huggingface(
        "Eyened/vascx:fovea/fovea_july24.pt"
    ).to(device)

    inputs = {"images": [item.to_serialized_dict() for item in data]}
    dataloader = ensemble_fovea._make_inference_dataloader(
        inputs,
        num_workers=8,
        preprocess=False,
        batch_size=8,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            items = ensemble_fovea._predict_batch(batch)
            items = [dataloader.dataset.transform.undo_item(item) for item in items]

            for item in items:
                yield item


def run_fovea_detection(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> pd.DataFrame:
    data = _create_dtos(rgb_paths, ce_paths=ce_paths, ids=ids)
    output_ids, outputs = [], []

    for item in iterate_fovea_detection(data, device=device):
        output_ids.append(item["id"])
        outputs.append(
            [
                *item["keypoints"][0].tolist(),
            ]
        )

    return pd.DataFrame(
        outputs,
        index=output_ids,
        columns=["x_fovea", "y_fovea"],
    )
