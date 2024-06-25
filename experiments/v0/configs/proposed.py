import copy
from experiments.v0.configs.base_configs import datamanager, trainer
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from models.proposed.model import UWNerfModelConfig

"""
Model Config
"""
model = UWNerfModelConfig(
    near_plane=0.05,
    far_plane=25.0,
    max_res=2048 * 256,
    use_distortion_loss=True,
    distortion_loss_mult=1e-4,
    use_transmittance_loss=True,
    transmittance_loss_mult=1e-4,
    transmittance_loss_beta_prior=10.0,
)

"""
Optimizer Config
"""
optimizer = AdamOptimizerConfig(
    lr=0.01,
    eps=1e-15,
    max_norm=None,
    weight_decay=0,
)
scheduler = ExponentialDecaySchedulerConfig(
    lr_pre_warmup=1e-08,
    lr_final=0.0001,
    warmup_steps=0,
    max_steps=200000,
    ramp="cosine",
)
pn_optimizer = AdamOptimizerConfig(
    lr=0.01,
    eps=1e-15,
    max_norm=None,
    weight_decay=0,
)
pn_scheduler = ExponentialDecaySchedulerConfig(
    lr_pre_warmup=1e-08,
    lr_final=0.0001,
    warmup_steps=0,
    max_steps=200000,
    ramp="cosine",
)

"""
Pipeline Config
"""
trainer = copy.deepcopy(trainer)
datamanager = copy.deepcopy(datamanager)
trainer.pipeline = VanillaPipelineConfig(datamanager=datamanager, model=model)

"""
Optimizer Config
"""
trainer.optimizers = {
    "fields": {
        "optimizer": optimizer,
        "scheduler": scheduler,
    },
    "proposal_networks": {
        "optimizer": pn_optimizer,
        "scheduler": pn_scheduler,
    },
}


"""
Other Training Configs
"""
trainer.method_name = "proposed"
trainer.load_dir = None
trainer.load_step = None
trainer.load_checkpoint = None
