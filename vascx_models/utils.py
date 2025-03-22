from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def create_fundus_overlay(
    rgb_path: str,
    av_path: Optional[str] = None,
    disc_path: Optional[str] = None,
    fovea_location: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a visualization of a fundus image with overlaid segmentations and markers.

    Args:
        rgb_path: Path to the RGB fundus image
        av_path: Optional path to artery-vein segmentation (1=artery, 2=vein, 3=intersection)
        disc_path: Optional path to binary disc segmentation
        fovea_location: Optional (x,y) tuple indicating the location of the fovea
        output_path: Optional path to save the visualization image

    Returns:
        Numpy array containing the visualization image
    """
    # Load RGB image
    rgb_img = np.array(Image.open(rgb_path))

    # Create output image starting with the RGB image
    output_img = rgb_img.copy()

    # Load and overlay AV segmentation if provided
    if av_path:
        av_mask = np.array(Image.open(av_path))

        # Create masks for arteries (1), veins (2) and intersections (3)
        artery_mask = av_mask == 1
        vein_mask = av_mask == 2
        intersection_mask = av_mask == 3

        # Combine artery and intersection for visualization
        artery_combined = np.logical_or(artery_mask, intersection_mask)
        vein_combined = np.logical_or(vein_mask, intersection_mask)

        # Apply colors: red for arteries, blue for veins
        # Red channel - increase for arteries
        output_img[artery_combined, 0] = 255
        output_img[artery_combined, 1] = 0
        output_img[artery_combined, 2] = 0

        # Blue channel - increase for veins
        output_img[vein_combined, 0] = 0
        output_img[vein_combined, 1] = 0
        output_img[vein_combined, 2] = 255

    # Load and overlay optic disc segmentation if provided
    if disc_path:
        disc_mask = np.array(Image.open(disc_path)) > 0

        # Apply white color for disc
        output_img[disc_mask, :] = [255, 255, 255]  # White

    # Convert to PIL image for drawing the fovea marker
    pil_img = Image.fromarray(output_img)

    # Add fovea marker if provided
    if fovea_location:
        draw = ImageDraw.Draw(pil_img)
        x, y = fovea_location
        marker_size = (
            min(pil_img.width, pil_img.height) // 50
        )  # Scale marker with image

        # Draw yellow X at fovea location
        draw.line(
            [(x - marker_size, y - marker_size), (x + marker_size, y + marker_size)],
            fill=(255, 255, 0),
            width=2,
        )
        draw.line(
            [(x - marker_size, y + marker_size), (x + marker_size, y - marker_size)],
            fill=(255, 255, 0),
            width=2,
        )

    # Convert back to numpy array
    output_img = np.array(pil_img)

    # Save output if path provided
    if output_path:
        Image.fromarray(output_img).save(output_path)

    return output_img


def batch_create_overlays(
    rgb_dir: Path,
    output_dir: Path,
    av_dir: Optional[Path] = None,
    disc_dir: Optional[Path] = None,
    fovea_data: Optional[Dict[str, Tuple[int, int]]] = None,
) -> None:
    """
    Create visualization overlays for a batch of images.

    Args:
        rgb_dir: Directory containing RGB fundus images
        output_dir: Directory to save visualization images
        av_dir: Optional directory containing AV segmentations
        disc_dir: Optional directory containing disc segmentations
        fovea_data: Optional dictionary mapping image IDs to fovea coordinates

    Returns:
        List of paths to created visualization images
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all RGB images
    rgb_files = list(rgb_dir.glob("*.png"))
    if not rgb_files:
        return []

    # Process each image
    for rgb_file in rgb_files:
        image_id = rgb_file.stem

        # Check for corresponding AV segmentation
        av_file = None
        if av_dir:
            av_file_path = av_dir / f"{image_id}.png"
            if av_file_path.exists():
                av_file = str(av_file_path)

        # Check for corresponding disc segmentation
        disc_file = None
        if disc_dir:
            disc_file_path = disc_dir / f"{image_id}.png"
            if disc_file_path.exists():
                disc_file = str(disc_file_path)

        # Get fovea location if available
        fovea_location = None
        if fovea_data and image_id in fovea_data:
            fovea_location = fovea_data[image_id]

        # Create output path
        output_file = output_dir / f"{image_id}.png"

        # Create and save overlay
        create_fundus_overlay(
            rgb_path=str(rgb_file),
            av_path=av_file,
            disc_path=disc_file,
            fovea_location=fovea_location,
            output_path=str(output_file),
        )
