from pathlib import Path

import click
import pandas as pd

from rtnls_fundusprep.cli import _run_preprocessing

from .inference import (
    run_fovea_detection,
    run_quality_estimation,
    run_segmentation_disc,
    run_segmentation_vessels_and_av,
)
from .utils import batch_create_overlays


@click.group(name="vascx")
def cli():
    pass


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--preprocess/--no-preprocess",
    default=True,
    help="Run preprocessing or use preprocessed images",
)
@click.option(
    "--vessels/--no-vessels", default=True, help="Run vessels and AV segmentation"
)
@click.option("--disc/--no-disc", default=True, help="Run optic disc segmentation")
@click.option(
    "--quality/--no-quality", default=True, help="Run image quality estimation"
)
@click.option("--fovea/--no-fovea", default=True, help="Run fovea detection")
@click.option(
    "--overlay/--no-overlay", default=True, help="Create visualization overlays"
)
@click.option("--n_jobs", type=int, default=4, help="Number of preprocessing workers")
@click.option(
    "--devices",
    default=None,
    help="Comma separated list of GPU ids to use (e.g., '0,1,2')",
)
@click.option(
    "--av_model",
    default="hf@Eyened/vascx:artery_vein/av_july24.pt",
    help="Model to use for artery-vein segmentation",
)
@click.option(
    "--vessels_model",
    default="hf@Eyened/vascx:vessels/vessels_july24.pt",
    help="Model to use for vessel segmentation",
)
@click.option(
    "--disc_model",
    default="hf@Eyened/vascx:disc/disc_july24.pt",
    help="Model to use for disc segmentation",
)
@click.option(
    "--quality_model",
    default="hf@Eyened/vascx:quality/quality.pt",
    help="Model to use for quality estimation",
)
@click.option(
    "--fovea_model",
    default="hf@Eyened/vascx:fovea/fovea_july24.pt",
    help="Model to use for fovea detection",
)
def run(
    data_path,
    output_path,
    preprocess,
    vessels,
    disc,
    quality,
    fovea,
    overlay,
    n_jobs,
    devices,
    av_model,
    vessels_model,
    disc_model,
    quality_model,
    fovea_model,
):
    """Run the complete inference pipeline on fundus images.

    DATA_PATH is either a directory containing images or a CSV file with 'path' column.
    OUTPUT_PATH is the directory where results will be stored.
    """

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Setup output directories
    preprocess_rgb_path = output_path / "preprocessed_rgb"
    vessels_path = output_path / "vessels"
    av_path = output_path / "artery_vein"
    disc_path = output_path / "disc"
    overlay_path = output_path / "overlays"

    # Parse devices option if provided
    device_list = None
    if devices:
        device_list = [int(d.strip()) for d in devices.split(",")]
        click.echo(f"Using GPUs: {device_list}")

    # Create required directories
    if preprocess:
        preprocess_rgb_path.mkdir(exist_ok=True, parents=True)
    if vessels:
        av_path.mkdir(exist_ok=True, parents=True)
        vessels_path.mkdir(exist_ok=True, parents=True)
    if disc:
        disc_path.mkdir(exist_ok=True, parents=True)
    if overlay:
        overlay_path.mkdir(exist_ok=True, parents=True)

    bounds_path = output_path / "bounds.csv" if preprocess else None
    quality_path = output_path / "quality.csv" if quality else None
    fovea_path = output_path / "fovea.csv" if fovea else None

    # Determine if input is a folder or CSV file
    data_path = Path(data_path)
    is_csv = data_path.suffix.lower() == ".csv"

    # Get files to process
    files = []
    ids = None

    if is_csv:
        click.echo(f"Reading file paths from CSV: {data_path}")
        try:
            df = pd.read_csv(data_path)
            if "path" not in df.columns:
                click.echo("Error: CSV must contain a 'path' column")
                return

            # Get file paths and convert to Path objects
            files = [Path(p) for p in df["path"]]

            if "id" in df.columns:
                ids = df["id"].tolist()
                click.echo("Using IDs from CSV 'id' column")

        except Exception as e:
            click.echo(f"Error reading CSV file: {e}")
            return
    else:
        click.echo(f"Finding files in directory: {data_path}")
        files = list(data_path.glob("*"))
        ids = [f.stem for f in files]

    if not files:
        click.echo("No files found to process")
        return

    click.echo(f"Found {len(files)} files to process")

    # Step 1: Preprocess images if requested
    if preprocess:
        click.echo("Running preprocessing...")
        _run_preprocessing(
            files=files,
            ids=ids,
            rgb_path=preprocess_rgb_path,
            bounds_path=bounds_path,
            n_jobs=n_jobs,
        )
        # Use the preprocessed images for subsequent steps
        preprocessed_files = list(preprocess_rgb_path.glob("*.png"))
    else:
        # Use the input files directly
        preprocessed_files = files
    ids = [f.stem for f in preprocessed_files]

    # Step 2: Run quality estimation if requested
    if quality:
        click.echo("Running quality estimation...")
        df_quality = run_quality_estimation(
            fpaths=preprocessed_files, ids=ids, devices=device_list, model=quality_model
        )
        df_quality.to_csv(quality_path)
        click.echo(f"Quality results saved to {quality_path}")

    # Step 3: Run vessels and AV segmentation if requested
    if vessels:
        click.echo("Running vessels and AV segmentation...")
        run_segmentation_vessels_and_av(
            rgb_paths=preprocessed_files,
            ids=ids,
            av_path=av_path,
            vessels_path=vessels_path,
            devices=device_list,
            av_model=av_model,
            vessels_model=vessels_model,
        )
        click.echo(f"Vessel segmentation saved to {vessels_path}")
        click.echo(f"AV segmentation saved to {av_path}")

    # Step 4: Run optic disc segmentation if requested
    if disc:
        click.echo("Running optic disc segmentation...")
        run_segmentation_disc(
            rgb_paths=preprocessed_files,
            ids=ids,
            output_path=disc_path,
            devices=device_list,
            model=disc_model,
        )
        click.echo(f"Disc segmentation saved to {disc_path}")

    # Step 5: Run fovea detection if requested
    df_fovea = None
    if fovea:
        click.echo("Running fovea detection...")
        df_fovea = run_fovea_detection(
            rgb_paths=preprocessed_files,
            ids=ids,
            devices=device_list,
            model=fovea_model,
        )
        df_fovea.to_csv(fovea_path)
        click.echo(f"Fovea detection results saved to {fovea_path}")

    # Step 6: Create overlays if requested
    if overlay:
        click.echo("Creating visualization overlays...")

        # Prepare fovea data if available
        fovea_data = None
        if df_fovea is not None:
            fovea_data = {
                idx: (row["x_fovea"], row["y_fovea"])
                for idx, row in df_fovea.iterrows()
            }

        # Create visualization overlays
        batch_create_overlays(
            rgb_dir=preprocess_rgb_path if preprocess else data_path,
            output_dir=overlay_path,
            av_dir=av_path,
            disc_dir=disc_path,
            fovea_data=fovea_data,
        )

        click.echo(f"Visualization overlays saved to {overlay_path}")

    click.echo(f"All requested processing complete. Results saved to {output_path}")
