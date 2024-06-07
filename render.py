from __future__ import annotations

from nerfstudio.utils.rich_utils import CONSOLE, ItersPerSecColumn
from nerfstudio.utils import colormaps, install_checks
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader
from nerfstudio.data.datamanagers.random_cameras_datamanager import (
    RandomCamerasDataManager,
)
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanagerConfig,
)
from nerfstudio.data.datasets.base_dataset import Dataset  # type: ignore
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.cameras.camera_paths import (
    get_interpolated_camera_path,
    get_spiral_path,
)
from rich.table import Table
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich import box, style

import numpy as np
import mediapy as media
from typing import List, Literal, Optional, Callable, Tuple
from dataclasses import dataclass, field
import gzip

import os
import sys
import torch
import yaml
import argparse
from pathlib import Path

from nerfstudio.configs.method_configs import all_methods
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint

from nerfstudio.scripts.render import (
    BaseRender,
    _disable_datamanager_setup,
    _render_trajectory_video,
)

from models.seathrunerf.seathrunerf_config import seathrunerf_method
from models.proposed.config import proposed_method


all_methods["proposed"] = proposed_method.config
all_methods["seathrunerf"] = seathrunerf_method.config


def eval_setup(
    config_path: Path,
    eval_num_rays_per_chunk: Optional[int] = None,
    test_mode: Literal["test", "val", "inference"] = "test",
    update_config_callback: Optional[Callable[[TrainerConfig], TrainerConfig]] = None,
) -> Tuple[TrainerConfig, Pipeline, Path, int]:
    """Shared setup for loading a saved pipeline for evaluation.

    Args:
        config_path: Path to config YAML file.
        eval_num_rays_per_chunk: Number of rays per forward pass
        test_mode:
            'val': loads train/val datasets into memory
            'test': loads train/test dataset into memory
            'inference': does not load any dataset into memory
        update_config_callback: Callback to update the config before loading the pipeline


    Returns:
        Loaded config, pipeline module, corresponding checkpoint, and step
    """
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, TrainerConfig)

    config.pipeline.datamanager._target = all_methods[
        config.method_name
    ].pipeline.datamanager._target
    if eval_num_rays_per_chunk:
        config.pipeline.model.eval_num_rays_per_chunk = eval_num_rays_per_chunk

    if update_config_callback is not None:
        config = update_config_callback(config)

    # load checkpoints from wherever they were saved
    config.load_dir = config.get_checkpoint_dir()

    # setup pipeline (which includes the DataManager)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = config.pipeline.setup(device=device, test_mode=test_mode)
    assert isinstance(pipeline, Pipeline)
    pipeline.eval()

    # load checkpointed information
    checkpoint_path, step = eval_load_checkpoint(config, pipeline)

    return config, pipeline, checkpoint_path, step


@dataclass
class DatasetRender(BaseRender):
    """Render all images in the dataset."""

    output_path: Path = Path("renders")
    """Path to output video file."""
    data: Optional[Path] = None
    """Override path to the dataset."""
    downscale_factor: Optional[float] = None
    """Scaling factor to apply to the camera image resolution."""
    split: Literal["train", "val", "test", "train+test"] = "test"
    """Split to render."""
    rendered_output_names: Optional[List[str]] = field(default_factory=lambda: None)
    """Name of the renderer outputs to use. rgb, depth, raw-depth, gt-rgb etc. By default all outputs are rendered."""

    def main(self):
        config: TrainerConfig

        def update_config(config: TrainerConfig) -> TrainerConfig:
            data_manager_config = config.pipeline.datamanager
            assert isinstance(
                data_manager_config,
                (VanillaDataManagerConfig, FullImageDatamanagerConfig),
            )
            data_manager_config.eval_num_images_to_sample_from = -1
            data_manager_config.eval_num_times_to_repeat_images = -1
            if isinstance(data_manager_config, VanillaDataManagerConfig):
                data_manager_config.train_num_images_to_sample_from = -1
                data_manager_config.train_num_times_to_repeat_images = -1
            if self.data is not None:
                data_manager_config.data = self.data
            if self.downscale_factor is not None:
                assert hasattr(data_manager_config.dataparser, "downscale_factor")
                setattr(
                    data_manager_config.dataparser,
                    "downscale_factor",
                    self.downscale_factor,
                )
            return config

        config, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="inference",
            update_config_callback=update_config,
        )
        data_manager_config = config.pipeline.datamanager
        assert isinstance(
            data_manager_config, (VanillaDataManagerConfig, FullImageDatamanagerConfig)
        )

        for split in self.split.split("+"):
            datamanager: VanillaDataManager
            dataset: Dataset
            if split == "train":
                with _disable_datamanager_setup(
                    data_manager_config._target  # type: ignore
                ):
                    datamanager = data_manager_config.setup(
                        test_mode="test", device=pipeline.device
                    )

                dataset = datamanager.train_dataset
                dataparser_outputs = getattr(
                    dataset, "_dataparser_outputs", datamanager.train_dataparser_outputs
                )
            else:
                with _disable_datamanager_setup(
                    data_manager_config._target  # type: ignore
                ):  # pylint: disable=protected-access
                    datamanager = data_manager_config.setup(
                        test_mode=split, device=pipeline.device
                    )

                dataset = datamanager.eval_dataset
                dataparser_outputs = getattr(dataset, "_dataparser_outputs", None)
                if dataparser_outputs is None:
                    dataparser_outputs = datamanager.dataparser.get_dataparser_outputs(
                        split=datamanager.test_split
                    )
            dataloader = FixedIndicesEvalDataloader(
                input_dataset=dataset,
                device=datamanager.device,
                num_workers=datamanager.world_size * 4,
            )
            images_root = Path(os.path.commonpath(dataparser_outputs.image_filenames))
            with Progress(
                TextColumn(f":movie_camera: Rendering split {split} :movie_camera:"),
                BarColumn(),
                TaskProgressColumn(
                    text_format="[progress.percentage]{task.completed}/{task.total:>.0f}({task.percentage:>3.1f}%)",
                    show_speed=True,
                ),
                ItersPerSecColumn(suffix="fps"),
                TimeRemainingColumn(elapsed_when_finished=False, compact=False),
                TimeElapsedColumn(),
            ) as progress:
                for camera_idx, (camera, batch) in enumerate(
                    progress.track(dataloader, total=len(dataset))
                ):
                    with torch.no_grad():
                        outputs = pipeline.model.get_outputs_for_camera(camera)

                    gt_batch = batch.copy()
                    gt_batch["rgb"] = gt_batch.pop("image")
                    all_outputs = (
                        list(outputs.keys())
                        + [f"raw-{x}" for x in outputs.keys()]
                        + [f"gt-{x}" for x in gt_batch.keys()]
                        + [f"raw-gt-{x}" for x in gt_batch.keys()]
                    )
                    rendered_output_names = self.rendered_output_names
                    if rendered_output_names is None:
                        rendered_output_names = ["gt-rgb"] + list(outputs.keys())
                    for rendered_output_name in rendered_output_names:
                        if rendered_output_name not in all_outputs:
                            CONSOLE.rule("Error", style="red")
                            CONSOLE.print(
                                f"Could not find {rendered_output_name} in the model outputs",
                                justify="center",
                            )
                            CONSOLE.print(
                                f"Please set --rendered-output-name to one of: {all_outputs}",
                                justify="center",
                            )
                            sys.exit(1)

                        is_raw = False
                        is_depth = rendered_output_name.find("depth") != -1
                        image_name = f"{camera_idx:05d}"

                        # Try to get the original filename
                        image_name = dataparser_outputs.image_filenames[
                            camera_idx
                        ].relative_to(images_root)

                        output_path = (
                            self.output_path / split / rendered_output_name / image_name
                        )
                        output_path.parent.mkdir(exist_ok=True, parents=True)

                        output_name = rendered_output_name
                        if output_name.startswith("raw-"):
                            output_name = output_name[4:]
                            is_raw = True
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                                if is_depth:
                                    # Divide by the dataparser scale factor
                                    output_image.div_(
                                        dataparser_outputs.dataparser_scale
                                    )
                        else:
                            if output_name.startswith("gt-"):
                                output_name = output_name[3:]
                                output_image = gt_batch[output_name]
                            else:
                                output_image = outputs[output_name]
                        del output_name

                        # Map to color spaces / numpy
                        if is_raw:
                            output_image = output_image.cpu().numpy()
                        elif is_depth:
                            output_image = (
                                colormaps.apply_depth_colormap(
                                    output_image,
                                    accumulation=outputs["accumulation"],
                                    near_plane=self.depth_near_plane,
                                    far_plane=self.depth_far_plane,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )
                        else:
                            output_image = (
                                colormaps.apply_colormap(
                                    image=output_image,
                                    colormap_options=self.colormap_options,
                                )
                                .cpu()
                                .numpy()
                            )

                        # Save to file
                        if is_raw:
                            with gzip.open(
                                output_path.with_suffix(".npy.gz"), "wb"
                            ) as f:
                                np.save(f, output_image)
                        elif self.image_format == "png":
                            media.write_image(
                                output_path.with_suffix(".png"), output_image, fmt="png"
                            )
                        elif self.image_format == "jpeg":
                            media.write_image(
                                output_path.with_suffix(".jpg"),
                                output_image,
                                fmt="jpeg",
                                quality=self.jpeg_quality,
                            )
                        else:
                            raise ValueError(
                                f"Unknown image format {self.image_format}"
                            )

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        for split in self.split.split("+"):
            table.add_row(f"Outputs {split}", str(self.output_path / split))
        CONSOLE.print(
            Panel(
                table,
                title="[bold][green]:tada: Render on split {} Complete :tada:[/bold]",
                expand=False,
            )
        )


class SpiralRender(BaseRender):
    """Render a spiral trajectory (often not great)."""

    seconds: float = 10.0
    """How long the video should be."""
    output_format: Literal["images", "video"] = "images"
    """How to save output data."""
    frame_rate: int = 5
    """Frame rate of the output video (only for interpolate trajectory)."""
    radius: float = 0.5
    """Radius of the spiral."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        install_checks.check_ffmpeg_installed()

        assert isinstance(
            pipeline.datamanager,
            (
                VanillaDataManager,
                ParallelDataManager,
                RandomCamerasDataManager,
            ),
        )
        steps = int(self.frame_rate * self.seconds)
        camera_start, _ = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0)
        camera_path = get_spiral_path(
            camera_start,
            steps=steps,
            radius=self.radius,
            zrate=0,
        )

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=self.seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )


@dataclass
class RenderInterpolated(BaseRender):
    """Render a trajectory that interpolates between training or eval dataset images."""

    pose_source: Literal["eval", "train"] = "eval"
    """Pose source to render."""
    interpolation_steps: int = 10
    """Number of interpolation steps between eval dataset cameras."""
    order_poses: bool = False
    """Whether to order camera poses by proximity."""
    frame_rate: int = 24
    """Frame rate of the output video."""
    output_format: Literal["images", "video"] = "images"
    """How to save output data."""

    def main(self) -> None:
        """Main function."""
        _, pipeline, _, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test",
        )

        install_checks.check_ffmpeg_installed()

        if self.pose_source == "eval":
            assert pipeline.datamanager.eval_dataset is not None
            cameras = pipeline.datamanager.eval_dataset.cameras
        else:
            assert pipeline.datamanager.train_dataset is not None
            cameras = pipeline.datamanager.train_dataset.cameras

        seconds = self.interpolation_steps * len(cameras) / self.frame_rate
        camera_path = get_interpolated_camera_path(
            cameras=cameras,
            steps=self.interpolation_steps,
            order_poses=self.order_poses,
        )

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            image_format=self.image_format,
            depth_near_plane=self.depth_near_plane,
            depth_far_plane=self.depth_far_plane,
            colormap_options=self.colormap_options,
            render_nearest_camera=self.render_nearest_camera,
            check_occlusions=self.check_occlusions,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--type",
        help="Spiral or Dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--load_config",
        help="Path to config YAML file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        help="render output path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--split",
        help="Split to render",
        type=str,
        default="train+test",
    )
    parser.add_argument(
        "--downscale_factor",
        help="Scaling factor to apply to the camera image resolution.",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    if args.type == "spiral":
        output_path = Path(
            args.output_path
            # if ".mp4" in args.output_path
            # else args.output_path + "spiral.mp4"
        )
        renderer = SpiralRender(
            load_config=Path(args.load_config),
            output_path=output_path,
            downscale_factor=int(args.downscale_factor),
            rendered_output_names=["restored"],
        )
    elif args.type == "interpolated":
        output_path = Path(
            args.output_path
            # if ".mp4" in args.output_path
            # else args.output_path + "interpolated.mp4"
        )
        renderer = RenderInterpolated(
            load_config=Path(args.load_config),
            output_path=output_path,
            downscale_factor=int(args.downscale_factor),
            rendered_output_names=["restored"],
        )
    elif args.type == "dataset":
        renderer = DatasetRender(
            load_config=Path(args.load_config),
            output_path=Path(args.output_path),
            split=args.split,
            downscale_factor=int(args.downscale_factor),
        )

    renderer.main()
