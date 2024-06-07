import torch
import os
import gc
import csv
import argparse
import datetime
from importlib import import_module

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
Define experiments to run
"""
experiments = {
    "nerfacto": {
        "v0": {
            "data_paths": [
                "datasets/Seathru-NeRF/Curasao/",
                "datasets/Seathru-NeRF/IUI3-RedSea/",
                "datasets/Seathru-NeRF/JapaneseGradens-RedSea/",
                "datasets/Seathru-NeRF/Panama/",
                "datasets/Eiffel-Tower/2015/",
                "datasets/Eiffel-Tower/2016/",
                "datasets/Eiffel-Tower/2018/",
                "datasets/Eiffel-Tower/2020/",
            ],
            "configs": [
                "nerfacto",
            ],
            "data_tags": [
                "Curasao",
                "IUI3-RedSea",
                "JapaneseGradens-RedSea",
                "Panama",
                "2015",
                "2016",
                "2018",
                "2020",
            ],
        },
    },
    "proposed": {
        "v0": {
            "data_paths": [
                "datasets/Seathru-NeRF/Curasao/",
                "datasets/Seathru-NeRF/IUI3-RedSea/",
                "datasets/Seathru-NeRF/JapaneseGradens-RedSea/",
                "datasets/Seathru-NeRF/Panama/",
                "datasets/Eiffel-Tower/2015/",
                "datasets/Eiffel-Tower/2016/",
                "datasets/Eiffel-Tower/2018/",
                "datasets/Eiffel-Tower/2020/",
            ],
            "configs": [
                "proposed",
            ],
            "data_tags": [
                "Curasao",
                "IUI3-RedSea",
                "JapaneseGradens-RedSea",
                "Panama",
                "2015",
                "2016",
                "2018",
                "2020",
            ],
        },
    },
    "seathrunerf": {
        "v0": {
            "data_paths": [
                "datasets/Seathru-NeRF/Curasao/",
                "datasets/Seathru-NeRF/IUI3-RedSea/",
                "datasets/Seathru-NeRF/JapaneseGradens-RedSea/",
                "datasets/Seathru-NeRF/Panama/",
                "datasets/Eiffel-Tower/2015/",
                "datasets/Eiffel-Tower/2016/",
                "datasets/Eiffel-Tower/2018/",
                "datasets/Eiffel-Tower/2020/",
            ],
            "configs": [
                "seathrunerf",
            ],
            "data_tags": [
                "Curasao",
                "IUI3-RedSea",
                "JapaneseGradens-RedSea",
                "Panama",
                "2015",
                "2016",
                "2018",
                "2020",
            ],
        },
    },
}


"""
Implement training setup
"""


def setup(exp_id):

    # Empty cache
    torch.cuda.empty_cache()

    # Get the setup date and time
    current_date = datetime.date.today().strftime("%d-%m-%Y")

    # Define the path to save summary files
    base_path = "logs/"
    error_log_path = os.path.join(base_path, "error.txt")
    summary_path = os.path.join(base_path, "success.csv")

    # Create summary files dir if it does not exist
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
    new_file = False
    if not os.path.exists(summary_path):
        new_file = True

    # Open and write summary
    summary_mode = "a" if os.path.exists(summary_path) else "w"
    error_mode = "a" if os.path.exists(summary_path) else "w"

    with open(summary_path, mode=summary_mode, newline="") as csvfile:
        # Metrics to log
        fieldnames = [
            "Training ID",
            "Experiment ID",
            "Data Description",
            "Data Path",
            "Method",
            "Start Time",
            "End Time",
            "Time Stamp",
        ]

        # Write Header
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        # Implement all experiments
        for experiment, experiment_info in experiments[exp_id].items():
            # Get all available and valid methods for this experiment
            methods_path = os.path.join("experiments", experiment, "configs")
            available_methods = os.listdir(methods_path)
            available_methods = [
                method
                for method in available_methods
                if ("__" not in method and "base" not in method)
            ]

            valid_methods = [
                method
                for method in experiment_info["configs"]
                if method + ".py" in available_methods
            ]

            # Implement method config file
            for method_name in valid_methods:
                # Get method config file
                trainer_config_path = f"experiments.{experiment}.configs.{method_name}"
                trainer_config = import_module(trainer_config_path).__getattribute__(
                    "trainer"
                )

                # Implement method for each data in data_paths
                for idx, data_path in enumerate(experiment_info["data_paths"]):
                    try:
                        # Train the model
                        trainer_config.pipeline.datamanager.data = data_path
                        trainer_config.experiment_name = (
                            "outputs/" + experiment_info["data_tags"][idx]
                        )
                        trainer_config.save_config()
                        trainer = trainer_config.setup()
                        trainer.setup()
                        trainer.train()
                        print(f"Training completed. Yay!!")

                        # Document training session outcome
                        train_summary = {
                            "Training ID": exp_id,
                            "Experiment ID": experiment,
                        }
                        train_summary["Data Description"] = experiment_info[
                            "data_tags"
                        ][idx]
                        train_summary["Data Path"] = data_path
                        train_summary["Method"] = method_name
                        train_summary["Start Time"] = str(datetime.datetime.now())
                        train_summary["Time Stamp"] = trainer_config.timestamp
                        train_summary["End Time"] = str(datetime.datetime.now())

                        writer.writerow(train_summary)

                    except Exception as e:
                        print(f"Training failed!! {e}")
                        num_gpus = torch.cuda.device_count()
                        t_memory = [
                            torch.cuda.get_device_properties(i).total_memory
                            for i in range(num_gpus)
                        ]
                        r_memory = [
                            torch.cuda.memory_reserved(i) for i in range(num_gpus)
                        ]
                        a_memory = [
                            torch.cuda.memory_allocated(i) for i in range(num_gpus)
                        ]
                        f_memory = [r_memory[i] - a_memory[i] for i in range(num_gpus)]

                        with open(error_log_path, mode=error_mode) as errorfile:
                            errorfile.write(f"   — Training DateTime: {current_date}\n")
                            errorfile.write(f"   — Training Status: Failed\n")
                            errorfile.write(f"   — Experiment: {exp_id}\n")
                            errorfile.write(f"   — Method: {method_name}\n")
                            errorfile.write(f"   — Data: {data_path}\n")
                            errorfile.write(f"   — Total Memory: {t_memory}\n")
                            errorfile.write(f"   — Reserved Memory: {r_memory}\n")
                            errorfile.write(f"   — Allocated Memory: {a_memory}\n")
                            errorfile.write(f"   — Free Memory: {f_memory}\n")
                            errorfile.write(f"   — Error: {str(e)}\n")
                            errorfile.write(
                                f"   — End Time: {datetime.datetime.now()}\n"
                            )
                            errorfile.write(f"\n")
                            errorfile.write(f"-" * 150)
                            errorfile.write(f"\n\n")
                        errorfile.close()

                    # Free GPU Memory
                    torch.cuda.empty_cache()

                # Free Other Memory
                try:
                    del trainer
                    del trainer_config
                    torch.cuda.empty_cache()
                    gc.collect()
                except:
                    pass

        csvfile.close()


# Implement setup file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "experiment_id",
        help="id of experiment to train",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Run experiment
    setup(args.experiment_id)
