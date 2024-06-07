"""
Code to train model.
"""

import os
import torch
import functools
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, cast


from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.utils import colormaps
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from utils import linear_to_srgb


class CustomTrainerConfig(TrainerConfig):
    """
    Custom Trainer Config Class to log additional details
    """

    # Target class to instantiate
    _target: Type = field(default_factory=lambda: CustomTrainer)

    # Whether to render images or not
    render_images: bool = True

    # Number of steps between renders
    steps_render_images: int = 1000

    # Whether to render train images alongside the eval images
    render_train_images: bool = True

    # Number of the evaluation images to render
    render_eval_images_size: int = 20

    # Number of the train images to render
    render_train_images_size: int = 20

    # Whether to rneder and save depth maps alongside the rgb images
    render_depth_maps: bool = False


class CustomTrainer(Trainer):
    """
    Custom Trainer class to log additional details
    """

    config: CustomTrainerConfig

    # @profiler.time_function
    def render_train_images(self, step: Optional[int] = None):
        """
        Renders the scene from selected training dataset camera poses

        Args:
            step: current training step
        """

        self.pipeline.eval()
        assert isinstance(self.pipeline.datamanager, VanillaDataManager)
        # Create a new data loader with fixed indices
        data_loader = FixedIndicesEvalDataloader(
            input_dataset=self.pipeline.datamanager.train_dataset,
            device=self.pipeline.datamanager.device,
            num_workers=self.pipeline.datamanager.world_size * 4,
        )

        # Get the total number of data in the dataset and select data indices
        # whose camera pose to render
        num_images = len(data_loader)
        if num_images > self.config.render_train_images_size:
            render_ids = np.linspace(
                int(0.05 * num_images),
                int(0.95 * num_images),
                self.config.render_train_images_size,
                dtype=int,
            ).tolist()
        else:
            render_ids = np.arange(0, num_images, 1, dtype=int).tolist()

        # Update the indices of the data to load
        data_loader.image_indices = render_ids

        # Render scene from selected camera poses and save rgb images and depth maps
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Rendering the scene from selected training dataset camera poses...",
                total=self.config.render_train_images_size,
            )

            # Create path to save the rgb images and depth maps
            rgb_output_path = self.base_dir / f"renders/train/rgb" / f"step-{step}"
            rgb_output_path.mkdir(parents=True, exist_ok=True)

            for camera_ray_bundle, batch in data_loader:
                # Get the image filename that corresponds to the camera pose to render
                image_filename = str(
                    self.pipeline.datamanager.train_dataset.image_filenames[
                        batch["image_idx"]
                    ]
                ).split("/")[-1]

                # Render scene and save pose rgb images
                outputs = self.pipeline.model.get_outputs_for_camera(camera_ray_bundle)

                # Get the rgb tensor from the outputs
                rgb = outputs["rgb"]
                # Convert tensors to images
                Image.fromarray((rgb * 255).byte().cpu().numpy()).save(
                    rgb_output_path / image_filename
                )

                # Render other images
                if "restored" in tuple(outputs.keys()):
                    restored = outputs["restored"]
                    output_path = (
                        self.base_dir / f"renders/train/restored" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray((restored * 255).byte().cpu().numpy()).save(
                        output_path / image_filename
                    )

                if "direct" in tuple(outputs.keys()):
                    direct = outputs["direct"]
                    output_path = (
                        self.base_dir / f"renders/train/direct" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray((direct * 255).byte().cpu().numpy()).save(
                        output_path / image_filename
                    )

                if "veiling_light" in tuple(outputs.keys()):
                    veiling_light = outputs["veiling_light"]
                    output_path = (
                        self.base_dir / f"renders/train/veiling_light" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray((veiling_light * 255).byte().cpu().numpy()).save(
                        output_path / image_filename
                    )

                if "backscatter" in tuple(outputs.keys()):
                    backscatter = outputs["backscatter"]
                    output_path = (
                        self.base_dir / f"renders/train/backscatter" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray((backscatter * 255).byte().cpu().numpy()).save(
                        output_path / image_filename
                    )

                if "object_backscatter" in tuple(outputs.keys()):
                    object_backscatter = outputs["object_backscatter"]
                    output_path = (
                        self.base_dir
                        / f"renders/train/object_backscatter"
                        / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray(
                        (object_backscatter * 255).byte().cpu().numpy()
                    ).save(output_path / image_filename)

                if "medium_backscatter" in tuple(outputs.keys()):
                    medium_backscatter = outputs["medium_backscatter"]
                    output_path = (
                        self.base_dir
                        / f"renders/train/medium_backscatter"
                        / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray(
                        (medium_backscatter * 255).byte().cpu().numpy()
                    ).save(output_path / image_filename)

                if "object_mask" in tuple(outputs.keys()):
                    output = outputs["object_mask"]
                    output_path = (
                        self.base_dir / f"renders/train/object_mask" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    if output.shape[-1] == 1:
                        Image.fromarray(
                            (output.squeeze(-1) * 255).byte().cpu().numpy(),
                            mode="L",
                        ).save(output_path / image_filename)
                    else:
                        Image.fromarray((output * 255).byte().cpu().numpy()).save(
                            output_path / image_filename
                        )

                if (
                    step == 0
                    or step == self.config.max_num_iterations // 2
                    or step == self.config.max_num_iterations - 1
                ):
                    if "densities" in tuple(outputs.keys()):
                        output = outputs["densities"]
                        output_path = (
                            self.base_dir / f"renders/train/densities" / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "steps" in tuple(outputs.keys()):
                        output = outputs["steps"]
                        output_path = (
                            self.base_dir / f"renders/train/steps" / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "object_transmittance" in tuple(outputs.keys()):
                        output = outputs["object_transmittance"]
                        output_path = (
                            self.base_dir
                            / f"renders/train/object_transmittance"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "object_alphas" in tuple(outputs.keys()):
                        output = outputs["object_alphas"]
                        output_path = (
                            self.base_dir
                            / f"renders/train/object_alphas"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "object_weights" in tuple(outputs.keys()):
                        output = outputs["object_weights"]
                        output_path = (
                            self.base_dir
                            / f"renders/train/object_weights"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "direct_coeffs" in tuple(outputs.keys()):
                        output = outputs["direct_coeffs"]
                        output_path = (
                            self.base_dir
                            / f"renders/train/direct_coeffs"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "backscatter_coeffs" in tuple(outputs.keys()):
                        output = outputs["backscatter_coeffs"]
                        output_path = (
                            self.base_dir
                            / f"renders/train/backscatter_coeffs"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                # Render scene and save pose depth maps
                if self.config.render_depth_maps:
                    median_depth_output_path = (
                        self.base_dir / f"renders/train/median_depths" / f"step-{step}"
                    )
                    expected_depth_output_path = (
                        self.base_dir
                        / f"renders/train/expected_depths"
                        / f"step-{step}"
                    )
                    if (
                        not median_depth_output_path.exists()
                        or not expected_depth_output_path.exists()
                    ):
                        median_depth_output_path.mkdir(parents=True, exist_ok=True)
                        expected_depth_output_path.mkdir(parents=True, exist_ok=True)

                    median_depth = colormaps.apply_depth_colormap(
                        outputs["depth"],
                        accumulation=outputs["accumulation"],
                    )

                    Image.fromarray((median_depth * 255).byte().cpu().numpy()).save(
                        median_depth_output_path / image_filename
                    )
                    file_path = os.path.splitext(image_filename)[0] + ".pt"
                    torch.save(outputs["depth"], median_depth_output_path / file_path)

                    expected_depth = colormaps.apply_depth_colormap(
                        outputs["expected_depth"],
                        accumulation=outputs["accumulation"],
                    )
                    Image.fromarray((expected_depth * 255).byte().cpu().numpy()).save(
                        expected_depth_output_path / image_filename
                    )
                    torch.save(
                        outputs["expected_depth"],
                        expected_depth_output_path / file_path,
                    )
                progress.advance(task)

        self.pipeline.train()

    # @profiler.time_function
    def render_eval_images(self, step: Optional[int] = None):
        """
        Renders the scene from selected training dataset camera poses

        Args:
            step: current training step
        """
        self.pipeline.eval()
        assert isinstance(self.pipeline.datamanager, VanillaDataManager)

        # Create a new data loader with fixed indices
        data_loader = self.pipeline.datamanager.fixed_indices_eval_dataloader

        # Get the total number of data in the dataset and select data indices
        # whose camera pose to render
        num_images = len(data_loader)
        if num_images > self.config.render_train_images_size:
            render_ids = np.linspace(
                int(0.05 * num_images),
                int(0.95 * num_images),
                self.config.render_train_images_size,
                dtype=int,
            ).tolist()
        else:
            render_ids = np.arange(0, num_images, 1, dtype=int).tolist()

        # Update the indices of the data to load
        data_loader.image_indices = render_ids

        # Render scene from selected camera poses and save rgb images and depth maps
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Rendering the scene from selected eval dataset camera poses...",
                total=self.config.render_eval_images_size,
            )

            # Render selected eval images
            # Create path to save the rgb and depth images
            rgb_output_path = self.base_dir / f"renders/eval/rgb" / f"step-{step}"
            rgb_output_path.mkdir(parents=True, exist_ok=True)

            for camera_ray_bundle, batch in data_loader:

                # Get the image filename that corresponds to the camera pose to render
                image_filename = str(
                    self.pipeline.datamanager.eval_dataset.image_filenames[
                        batch["image_idx"]
                    ]
                ).split("/")[-1]

                # Render scene and save pose rgb images
                outputs = self.pipeline.model.get_outputs_for_camera(camera_ray_bundle)

                # Get the rgb tensor from the outputs
                rgb = outputs["rgb"]
                # Convert tensors to images
                Image.fromarray((rgb * 255).byte().cpu().numpy()).save(
                    rgb_output_path / image_filename
                )

                # Render other images
                if "direct" in tuple(outputs.keys()):
                    direct = outputs["direct"]
                    output_path = (
                        self.base_dir / f"renders/eval/direct" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray((direct * 255).byte().cpu().numpy()).save(
                        output_path / image_filename
                    )

                if "restored" in tuple(outputs.keys()):
                    restored = outputs["restored"]
                    output_path = (
                        self.base_dir / f"renders/eval/restored" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray((restored * 255).byte().cpu().numpy()).save(
                        output_path / image_filename
                    )
                    balanced_restored = linear_to_srgb(restored)
                    balanced_output_path = (
                        self.base_dir
                        / f"renders/eval/restored_balanced"
                        / f"step-{step}"
                    )
                    if not balanced_output_path.exists():
                        balanced_output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray(
                        (balanced_restored * 255).byte().cpu().numpy()
                    ).save(balanced_output_path / image_filename)

                if "veiling_light" in tuple(outputs.keys()):
                    veiling_light = outputs["veiling_light"]
                    output_path = (
                        self.base_dir / f"renders/eval/veiling_light" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray((veiling_light * 255).byte().cpu().numpy()).save(
                        output_path / image_filename
                    )

                if "backscatter" in tuple(outputs.keys()):
                    backscatter = outputs["backscatter"]
                    output_path = (
                        self.base_dir / f"renders/eval/backscatter" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray((backscatter * 255).byte().cpu().numpy()).save(
                        output_path / image_filename
                    )

                if "object_backscatter" in tuple(outputs.keys()):
                    object_backscatter = outputs["object_backscatter"]
                    output_path = (
                        self.base_dir
                        / f"renders/eval/object_backscatter"
                        / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray(
                        (object_backscatter * 255).byte().cpu().numpy()
                    ).save(output_path / image_filename)

                if "medium_backscatter" in tuple(outputs.keys()):
                    medium_backscatter = outputs["medium_backscatter"]
                    output_path = (
                        self.base_dir
                        / f"renders/eval/medium_backscatter"
                        / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    Image.fromarray(
                        (medium_backscatter * 255).byte().cpu().numpy()
                    ).save(output_path / image_filename)

                if "object_mask" in tuple(outputs.keys()):
                    output = outputs["object_mask"]
                    output_path = (
                        self.base_dir / f"renders/eval/object_mask" / f"step-{step}"
                    )
                    if not output_path.exists():
                        output_path.mkdir(parents=True, exist_ok=True)

                    if output.shape[-1] == 1:
                        Image.fromarray(
                            (output.squeeze(-1) * 255).byte().cpu().numpy(),
                            mode="L",
                        ).save(output_path / image_filename)
                    else:
                        Image.fromarray((output * 255).byte().cpu().numpy()).save(
                            output_path / image_filename
                        )

                if (
                    step == 0
                    or step == self.config.max_num_iterations // 2
                    or step == self.config.max_num_iterations - 1
                ):
                    if "densities" in tuple(outputs.keys()):
                        output = outputs["densities"]
                        output_path = (
                            self.base_dir / f"renders/eval/densities" / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "steps" in tuple(outputs.keys()):
                        output = outputs["steps"]
                        output_path = (
                            self.base_dir / f"renders/eval/steps" / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "object_transmittance" in tuple(outputs.keys()):
                        output = outputs["object_transmittance"]
                        output_path = (
                            self.base_dir
                            / f"renders/eval/object_transmittance"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "object_alphas" in tuple(outputs.keys()):
                        output = outputs["object_alphas"]
                        output_path = (
                            self.base_dir
                            / f"renders/eval/object_alphas"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "object_weights" in tuple(outputs.keys()):
                        output = outputs["object_weights"]
                        output_path = (
                            self.base_dir
                            / f"renders/eval/object_weights"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "direct_coeffs" in tuple(outputs.keys()):
                        output = outputs["direct_coeffs"]
                        output_path = (
                            self.base_dir
                            / f"renders/eval/direct_coeffs"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                    if "backscatter_coeffs" in tuple(outputs.keys()):
                        output = outputs["backscatter_coeffs"]
                        output_path = (
                            self.base_dir
                            / f"renders/eval/backscatter_coeffs"
                            / f"step-{step}"
                        )
                        if not output_path.exists():
                            output_path.mkdir(parents=True, exist_ok=True)
                        file_path = os.path.splitext(image_filename)[0] + ".pt"
                        torch.save(output, output_path / file_path)

                # Render scene and save pose depth maps

                if self.config.render_depth_maps:
                    median_depth_output_path = (
                        self.base_dir / f"renders/eval/median_depths" / f"step-{step}"
                    )
                    expected_depth_output_path = (
                        self.base_dir / f"renders/eval/expected_depths" / f"step-{step}"
                    )
                    if (
                        not median_depth_output_path.exists()
                        or not expected_depth_output_path.exists()
                    ):
                        median_depth_output_path.mkdir(parents=True, exist_ok=True)
                        expected_depth_output_path.mkdir(parents=True, exist_ok=True)

                    median_depth = colormaps.apply_depth_colormap(
                        outputs["depth"],
                        accumulation=outputs["accumulation"],
                    )
                    Image.fromarray((median_depth * 255).byte().cpu().numpy()).save(
                        median_depth_output_path / image_filename
                    )
                    file_path = os.path.splitext(image_filename)[0] + ".pt"
                    torch.save(outputs["depth"], median_depth_output_path / file_path)

                    expected_depth = colormaps.apply_depth_colormap(
                        outputs["expected_depth"],
                        accumulation=outputs["accumulation"],
                    )
                    Image.fromarray((expected_depth * 255).byte().cpu().numpy()).save(
                        expected_depth_output_path / image_filename
                    )
                    torch.save(outputs["depth"], expected_depth_output_path / file_path)
                progress.advance(task)

        self.pipeline.train()

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(
                step=step
            )
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(
                name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step
            )
            writer.put_dict(
                name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step
            )

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = (
                    self.pipeline.get_eval_image_metrics_and_images(step=step)
                )
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(
                name="Eval Images Metrics", scalar_dict=metrics_dict, step=step
            )
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if (
            step_check(step, self.config.steps_per_eval_all_images)
            or step == self.config.max_num_iterations - 1
        ):
            metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(
                name="Eval Images Metrics Dict (all images)",
                scalar_dict=metrics_dict,
                step=step,
            )

        # Render the scene from selected dataset camera poses
        if self.config.render_images:
            if (
                step == 0
                or step_check(step, self.config.steps_render_images)
                or step == self.config.max_num_iterations - 1
            ):

                if self.config.render_train_images:
                    self.render_train_images(step)
                self.render_eval_images(step)
