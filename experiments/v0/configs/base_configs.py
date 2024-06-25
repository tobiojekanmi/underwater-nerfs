from nerfstudio.configs.base_config import (
    LocalWriterConfig,
    LoggingConfig,
    MachineConfig,
    ViewerConfig,
)
from pathlib import Path, PosixPath
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from engine.trainer import CustomTrainer, CustomTrainerConfig


"""
Base Trainer Config
"""
trainer = CustomTrainerConfig()
trainer._target = CustomTrainer

"""
Writer and Logging Config
"""
local_writer = LocalWriterConfig(max_log_size=1000, enable=True)
trainer.logging = LoggingConfig(
    steps_per_log=250,
    max_buffer_size=20,
    local_writer=local_writer,
    profiler="pytorch",
)


"""
Machine Config
"""
trainer.machine = MachineConfig(
    seed=42,
    num_devices=1,
    num_machines=1,
    machine_rank=0,
    dist_url="auto",
    device_type="cuda",
)


"""
Viewer Config
"""
trainer.viewer = ViewerConfig(
    relative_log_filename="viewer_log_filename.txt",
    websocket_port=40019,
    websocket_port_default=7007,
    websocket_host="0.0.0.0",
    num_rays_per_chunk=2**15,
    max_num_display_images=128,
    quit_on_train_completion=True,
    image_format="jpeg",
    jpeg_quality=75,
    make_share_url=False,
    default_composite_depth=True,
)


"""
Data Parser and Data Manager Config
"""
dataparser = ColmapDataParserConfig(
    data=PosixPath("datasets/Eiffel-Tower/2015"),
    scale_factor=1.0,
    downscale_factor=2,
    scene_scale=2.0,
    orientation_method="up",
    center_method="poses",
    auto_scale_poses=True,
    train_split_fraction=0.95,
    depth_unit_scale_factor=1e-3,
    images_path=PosixPath("images/"),
    masks_path=None,
    depths_path=None,
    colmap_path=PosixPath("sparse/0"),
    load_3D_points=False,
    max_2D_matches_per_3D_point=0,
)

datamanager = VanillaDataManagerConfig(
    dataparser=dataparser,
    train_num_rays_per_batch=4096 * 3,
    train_num_images_to_sample_from=-1,
    train_num_times_to_repeat_images=-1,
    eval_num_rays_per_batch=4096 * 3,
    eval_num_images_to_sample_from=-1,
    eval_num_times_to_repeat_images=-1,
    eval_image_indices=None,
    patch_size=1,
    images_on_gpu=False,
    masks_on_gpu=False,
)


"""
Other Basic Trainer Configs
"""
trainer.output_dir = Path("experiments/v0/")
trainer.experiment_name = "outputs"
trainer.project_name = None
trainer.timestamp = "{timestamp}"
trainer.set_timestamp()
trainer.vis = "viewer+tensorboard"
trainer.relative_model_dir = Path("models/")
trainer.steps_per_save = 2000
trainer.steps_per_eval_batch = 1000
trainer.steps_per_eval_image = 1000
trainer.steps_per_eval_all_images = 10000
trainer.max_num_iterations = 100000
trainer.mixed_precision = True
trainer.use_grad_scaler = False
trainer.save_only_latest_checkpoint = True
trainer.log_gradients = False


"""
Rendering Configs
"""
trainer.steps_render_images = 1000
trainer.render_train_images = True
trainer.render_eval_images_size = 5
trainer.render_train_images_size = 10
trainer.render_depth_maps = True
