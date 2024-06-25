import copy
from .base_configs import datamanager, trainer
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.models.nerfacto import NerfactoModelConfig


"""
Model Config
"""
model = NerfactoModelConfig(
    near_plane=0.05,
    far_plane=25,
    background_color="random",
    hidden_dim=128,
    hidden_dim_color=256,
    num_levels=16,
    base_res=16,
    max_res=2048 * 256,
    log2_hashmap_size=19,
    features_per_level=2,
    num_proposal_samples_per_ray=(256, 128),
    num_nerf_samples_per_ray=64,
    proposal_update_every=5,
    proposal_warmup=5000,
    num_proposal_iterations=2,
    use_same_proposal_network=False,
    proposal_net_args_list=[
        {
            "hidden_dim": 16,
            "log2_hashmap_size": 17,
            "num_levels": 5,
            "max_res": 128,
            "use_linear": False,
        },
        {
            "hidden_dim": 16,
            "log2_hashmap_size": 17,
            "num_levels": 5,
            "max_res": 256,
            "use_linear": False,
        },
    ],
    proposal_initial_sampler="piecewise",
    interlevel_loss_mult=1.0,
    distortion_loss_mult=0.002,
    orientation_loss_mult=0.0001,
    pred_normal_loss_mult=0.001,
    use_proposal_weight_anneal=True,
    use_average_appearance_embedding=True,
    proposal_weights_anneal_slope=10.0,
    proposal_weights_anneal_max_num_iters=1000,
    use_single_jitter=True,
    predict_normals=False,
    disable_scene_contraction=False,
    use_gradient_scaling=False,
    implementation="tcnn",
    appearance_embed_dim=32,
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
trainer.method_name = "nerfacto"
trainer.load_dir = None
trainer.load_step = None
trainer.load_checkpoint = None
