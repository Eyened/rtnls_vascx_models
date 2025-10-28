import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from rtnls_inference.ensembles.ensemble_classification import ClassificationEnsemble
from rtnls_inference.ensembles.ensemble_heatmap_regression import (
    HeatmapRegressionEnsemble,
)
from rtnls_inference.ensembles.ensemble_segmentation import SegmentationEnsemble
from rtnls_inference.utils import decollate_batch, extract_keypoints_from_heatmaps


def run_quality_estimation(fpaths, ids, device: torch.device):
    ensemble_quality = ClassificationEnsemble.from_release("quality.pt").to(device)
    dataloader = ensemble_quality._make_inference_dataloader(
        fpaths,
        ids=ids,
        num_workers=8,
        preprocess=False,
        batch_size=16,
    )

    output_ids, outputs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            im = batch["image"].to(device)

            # QUALITY
            quality = ensemble_quality.predict_step(im)
            quality = torch.mean(quality, dim=0)

            items = {"id": batch["id"], "quality": quality}
            items = decollate_batch(items)

            for item in items:
                output_ids.append(item["id"])
                outputs.append(item["quality"].tolist())

    return pd.DataFrame(
        outputs,
        index=output_ids,
        columns=["q1", "q2", "q3"],
    )


def run_segmentation_vessels_and_av(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    av_path: Optional[Path] = None,
    vessels_path: Optional[Path] = None,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
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
    """
    # Create output directories if they don't exist
    if av_path is not None:
        av_path.mkdir(exist_ok=True, parents=True)
    if vessels_path is not None:
        vessels_path.mkdir(exist_ok=True, parents=True)

    # Load models
    ensemble_av = SegmentationEnsemble.from_release("av_july24.pt").to(device).eval()
    ensemble_vessels = (
        SegmentationEnsemble.from_release("vessels_july24.pt").to(device).eval()
    )

    # Prepare input paths
    if ce_paths is None:
        # If CE paths are not provided, use RGB paths for both inputs
        fpaths = rgb_paths
    else:
        # If CE paths are provided, pair them with RGB paths
        if len(rgb_paths) != len(ce_paths):
            raise ValueError("rgb_paths and ce_paths must have the same length")
        fpaths = list(zip(rgb_paths, ce_paths))

    # Create dataloader
    dataloader = ensemble_av._make_inference_dataloader(
        fpaths,
        ids=ids,
        num_workers=8,
        preprocess=False,
        batch_size=8,
    )

    # Run inference
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # AV segmentation
            if av_path is not None:
                with torch.autocast(device_type=device.type):
                    proba = ensemble_av.forward(batch["image"].to(device))
                proba = torch.mean(proba, dim=1)  # average over models
                proba = torch.permute(proba, (0, 2, 3, 1))  # NCHW -> NHWC
                proba = torch.nn.functional.softmax(proba, dim=-1)

                items = {
                    "id": batch["id"],
                    "image": proba,
                }

                items = decollate_batch(items)
                for i, item in enumerate(items):
                    fpath = os.path.join(av_path, f"{item['id']}.png")
                    mask = np.argmax(item["image"], -1)
                    Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)

            # Vessel segmentation
            if vessels_path is not None:
                with torch.autocast(device_type=device.type):
                    proba = ensemble_vessels.forward(batch["image"].to(device))
                proba = torch.mean(proba, dim=1)  # average over models
                proba = torch.permute(proba, (0, 2, 3, 1))  # NCHW -> NHWC
                proba = torch.nn.functional.softmax(proba, dim=-1)

                items = {
                    "id": batch["id"],
                    "image": proba,
                }

                items = decollate_batch(items)
                for i, item in enumerate(items):
                    fpath = os.path.join(vessels_path, f"{item['id']}.png")
                    mask = np.argmax(item["image"], -1)
                    Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)


def run_segmentation_disc(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> None:
    ensemble_disc = (
        SegmentationEnsemble.from_release("disc_july24.pt").to(device).eval()
    )

    # Prepare input paths
    if ce_paths is None:
        # If CE paths are not provided, use RGB paths for both inputs
        fpaths = rgb_paths
    else:
        # If CE paths are provided, pair them with RGB paths
        if len(rgb_paths) != len(ce_paths):
            raise ValueError("rgb_paths and ce_paths must have the same length")
        fpaths = list(zip(rgb_paths, ce_paths))

    dataloader = ensemble_disc._make_inference_dataloader(
        fpaths,
        ids=ids,
        num_workers=8,
        preprocess=False,
        batch_size=8,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # AV
            with torch.autocast(device_type=device.type):
                proba = ensemble_disc.forward(batch["image"].to(device))
            proba = torch.mean(proba, dim=1)  # average over models
            proba = torch.permute(proba, (0, 2, 3, 1))  # NCHW -> NHWC
            proba = torch.nn.functional.softmax(proba, dim=-1)

            items = {
                "id": batch["id"],
                "image": proba,
            }

            items = decollate_batch(items)
            items = [dataloader.dataset.transform.undo_item(item) for item in items]
            for i, item in enumerate(items):
                fpath = os.path.join(output_path, f"{item['id']}.png")

                mask = np.argmax(item["image"], -1)
                Image.fromarray(mask.squeeze().astype(np.uint8)).save(fpath)


def run_fovea_detection(
    rgb_paths: List[Path],
    ce_paths: Optional[List[Path]] = None,
    ids: Optional[List[str]] = None,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
) -> None:
    # def run_fovea_detection(fpaths, ids, device: torch.device):
    ensemble_fovea = HeatmapRegressionEnsemble.from_release("fovea_july24.pt").to(
        device
    )

    # Prepare input paths
    if ce_paths is None:
        # If CE paths are not provided, use RGB paths for both inputs
        fpaths = rgb_paths
    else:
        # If CE paths are provided, pair them with RGB paths
        if len(rgb_paths) != len(ce_paths):
            raise ValueError("rgb_paths and ce_paths must have the same length")
        fpaths = list(zip(rgb_paths, ce_paths))

    dataloader = ensemble_fovea._make_inference_dataloader(
        fpaths,
        ids=ids,
        num_workers=8,
        preprocess=False,
        batch_size=8,
    )

    output_ids, outputs = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if len(batch) == 0:
                continue

            im = batch["image"].to(device)

            # FOVEA DETECTION
            with torch.autocast(device_type=device.type):
                heatmap = ensemble_fovea.forward(im)
            keypoints = extract_keypoints_from_heatmaps(heatmap)

            kp_fovea = torch.mean(keypoints, dim=1)  # average over models

            items = {
                "id": batch["id"],
                "keypoints": kp_fovea,
                "metadata": batch["metadata"],
            }
            items = decollate_batch(items)

            items = [dataloader.dataset.transform.undo_item(item) for item in items]

            for item in items:
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
