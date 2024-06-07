from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from .model import SeathruNerfModelConfig

# Base method configuration
seathrunerf_method = MethodSpecification(
    config=TrainerConfig(
        method_name="seathru-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=100000,
        mixed_precision=True,
        pipeline=VanillaPipelineConfig(
            datamanager=VanillaDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(),
                train_num_rays_per_batch=8192,
                eval_num_rays_per_batch=4096,
            ),
            model=SeathruNerfModelConfig(eval_num_rays_per_chunk=1 << 15),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-8, max_norm=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-5, max_steps=500000, warmup_steps=1024
                ),
            },
            "camera_opt": {
                "mode": "off",
                "optimizer": AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=6e-6, max_steps=500000
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Seathru-NeRF",
)
