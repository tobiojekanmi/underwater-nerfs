"""
The datasetup.py script enables you to download and preprocess 
your COLMAP data to a nerfstudio format. 

Consult the ReadMe file on how to use the script. 
"""

import argparse
import os
import sys
import tqdm
import shutil
import subprocess
from pathlib import Path
from nerfstudio.process_data.colmap_utils import create_sfm_depth
from nerfstudio.data.utils.colmap_parsing_utils import read_images_binary
from nerfstudio.process_data.process_data_utils import downscale_images
from nerfstudio.process_data.colmap_utils import colmap_to_json
from typing import Optional
from collections import namedtuple
from torchvision.datasets.utils import download_and_extract_archive


# Supported datasets
datasets = {"Eiffel-Tower": ["2015", "2016", "2018", "2020"]}

# Named tuple to store batch information
batch = namedtuple("batch", ["url", "dataset"])

# Dictionary to store batch information for each year datasets
batches = {
    "2015": batch(
        url="https://www.seanoe.org/data/00810/92226/data/98240.zip",
        dataset="Eiffel-Tower",
    ),
    "2016": batch(
        url="https://www.seanoe.org/data/00810/92226/data/98289.zip",
        dataset="Eiffel-Tower",
    ),
    "2018": batch(
        url="https://www.seanoe.org/data/00810/92226/data/98314.zip",
        dataset="Eiffel-Tower",
    ),
    "2020": batch(
        url="https://www.seanoe.org/data/00810/92226/data/98356.zip",
        dataset="Eiffel-Tower",
    ),
    "custom": batch(
        url=None,
        dataset="custom",
    ),
}


def setup_dataset(
    dataset_name: str,
    dataset_url: str = Optional[None],
    download_root: str = Optional[None],
    delete_archive: bool = True,
):
    """
    Downloads and extracts the dataset of interest

    Args:
        dataset_name: name of the dataset or batch of the dataset to
            download (for supported dataset like the Eiffel Tower, specify
            'Eiffel-Tower' to download data for all years or specify
            only the particular year/batch of interest. (e.g. 2015).
            To download from url, specify 'custom')
        dataset_url: url of the dataset. dataset_name must be 'custom'
        delete_archive: deletes archive after extraction if True.
            Keep otherwise.
    """

    # Get specific dataset batch to download from the list of supported batches
    dataset = batches[dataset_name]

    # Set the dataset url for custom data
    if dataset_name == "custom":
        if dataset_url is not None:
            dataset = dataset._replace(url=dataset_url)
        else:
            raise ValueError("custom dataset url cannot be None")

    # Set the download root dir name
    if download_root is None:
        download_root = dataset.dataset

    # Download and extract the archive
    if not delete_archive:
        download_and_extract_archive(dataset.url, download_root)
    else:
        download_and_extract_archive(dataset.url, download_root, remove_finished=True)


def train_test_split(
    images_path: str,
    colmap_model_path: str,
    output_path: str,
    test_size: int or float = None,
    split_strategy: str = None,
    start_test_data_from: int or float = None,
):
    """
    Splits the images in the dataset_path into train and test sets
    using the split_strategy provided

    Args:
        images_path: path to the directory containing the images
            of interest
        colmap_model_path: path to the directory containing the images,
            cameras, and points3D files. These files can be in text (.txt)
            or binary (.bin) format.
        output_path: path to the directory to store train and test data
        test_size: proportion of the dataset to include in the test dataset.
            For integer values, test_size images are used while
            test_size*data_size images are used for float values
        split_strategy: one of `interleaved` or `consecutive`
        start_test_data_from: where to start test data from when
            split_strategy is consecutive. int values represent exact id
            while float represent the whole data proportion id (i.e., 0.5 means
             from the middle id)
    """

    # Aggragate and sort image filenames
    image_filenames = os.listdir(images_path)
    image_filenames.sort()

    # Get test dataset size
    data_size = len(image_filenames)
    test_size = int(test_size * data_size if 0 < test_size < 1 else test_size)

    # Split image filenames into train and test sets using a strategy
    if split_strategy.lower() == "interleaved":
        test_data_ids = range(0, data_size, int(data_size / test_size))[:test_size]
        test_data_filenames = [image_filenames[idx] for idx in test_data_ids]
        train_data_filenames = [
            image_filenames[idx]
            for idx in range(data_size)
            if image_filenames[idx] not in test_data_filenames
        ]

    elif split_strategy.lower() == "consecutive":
        if start_test_data_from is not None:
            start_split_id = int(
                start_test_data_from * data_size
                if 0 < start_test_data_from < 1
                else start_test_data_from
            )
        else:
            start_split_id = 0

        end_split_id = start_split_id + test_size
        test_data_filenames = image_filenames[start_split_id:end_split_id]
        train_data_filenames = [
            image_filenames[idx]
            for idx in range(data_size)
            if image_filenames[idx] not in test_data_filenames
        ]
    else:
        raise ValueError(
            f"split_strategy must be one of `interleaved` or `consecutive`."
            f"{split_strategy} given"
        )

    # Create new folder to store these datasets
    for directory in ["train", "test", "train/images", "test/images"]:
        if not os.path.isdir(os.path.join(output_path, directory)):
            os.mkdir(os.path.join(output_path, directory))

    # Save the image filenames in txt files
    # (Note: the train image filenames will be stored in the test directory
    # as images to delete from the model)
    for image_filenames, file_dst_path in zip(
        [test_data_filenames, train_data_filenames],
        [
            os.path.join(output_path, "train/images_removed.txt"),
            os.path.join(output_path, "test/images_removed.txt"),
        ],
    ):
        with open(file_dst_path, "w") as file:
            for item in image_filenames:
                file.write("%s\n" % item)
            file.close()

    # Implement data splitting
    print("New Dataset Summary")
    print("Total data size: ", data_size)
    print("Train data size: ", data_size - test_size)
    print("Test data size: ", test_size, "\n")

    for image_filenames, directory in zip(
        [train_data_filenames, test_data_filenames], ["train", "test"]
    ):
        print(f"--------- Creating {directory} dataset ----------")
        # Create new COLMAP model files for each dataset
        print(f"Creating {directory} colmap model files")
        del_files_list_path = os.path.join(output_path, directory, "images_removed.txt")
        model_dst_path = os.path.join(output_path, directory)
        del_data_command = " ".join(
            [
                "colmap",
                "image_deleter",
                "--input_path",
                colmap_model_path,
                "--output_path",
                model_dst_path,
                "--image_names_path",
                del_files_list_path,
            ]
        )
        _ = run_command(del_data_command)
        print("Done!!\n")

        # Move the dataset images to its output path
        print(f"Moving {directory} images")
        for filename in tqdm.tqdm(image_filenames):
            src_path = os.path.join(images_path, filename)
            dst_path = os.path.join(output_path, directory, "images", filename)
            shutil.move(src_path, dst_path)

        print("Done!!\n")

    print("---- Removing residual folders/files (if any) ----")
    if images_path != output_path:
        run_command("rm -r " + images_path)
    if colmap_model_path != output_path:
        run_command("rm -r " + colmap_model_path)
    print("Done!!\n")


def create_colmap_bin_files(model_path):
    """
    Creates the binary equivalent of the model txt files

    Args:
        model_path: path to the model_files
    """

    model_files = os.listdir(model_path)

    # Create binary model files if one or more binary files does not exist
    if not all(
        file in model_files for file in ["images.bin", "images.bin", "images.bin"]
    ):
        print(f"--------- Creating binary model files ----------")
        create_bin_command = " ".join(
            [
                "colmap",
                "model_converter",
                "--input_path",
                model_path,
                "--output_path",
                model_path,
                "--output_type",
                "BIN",
            ]
        )
        _ = run_command(create_bin_command)
        print("Done!!\n")


def run_command(command: str) -> Optional[str]:
    """
    Runs a command and returns the output.

    Args:
        command: command to run.
    """

    # Run command
    out = subprocess.run(command, capture_output=True, shell=True)

    # Display error if any
    if out.returncode != 0:
        print(f"Error running command: {command}")
        print(out.stderr.decode("utf-8"))
        sys.exit(1)

    # Return output if the is any
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


# Implement datasetup file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "action",
        choices=[
            "download",
            "split_data",
            "create_colmap_bin_files",
            "downscale_images",
            "create_sfm_depths",
        ],
        help="setup action to implement",
    )
    parser.add_argument(
        "--dataset",
        choices=list(datasets.keys()) + list(batches.keys()) + ["custom"],
        help="name or url of the dataset or single batch to setup",
        default="custom",
    )
    parser.add_argument(
        "--dataset_url",
        help="for custom data, specify url of the data",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--download_root",
        help="name of the directory to download and extract archive to",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--delete_archive",
        help="add flag to delete archive after extracting it",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--downscale_factor",
        help="downscale factor (number of times to downscale images)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--images_path",
        help="images directory path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--colmap_model_path",
        help="colmap model directory path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        help="output directory path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sfm_depths_path",
        help="depth maps path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_size",
        help="proportion of the dataset to include in the test split. \
            (must be a float in the range of 0 and 1)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--split_strategy",
        help="technique for splitting the data (one of interleaved or consecutive).",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_split_from",
        help="where to start splitting the dataset. \
            (must be a float in the range of 0 and 1)",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    # Implement Module
    base_data_path = os.path.join(os.path.dirname(__file__))
    
    if args.action == "download":
        print("-" * 75)
        print(f"Downloading dataset")
        print("-" * 75)

        if args.dataset in datasets:
            for batch_name in datasets[args.dataset]:
                if args.images_path is None:
                    args.images_path = os.path.join(
                        base_data_path, args.dataset, batch_name, "images"
                    )
                if args.colmap_model_path is None:
                    args.colmap_model_path = os.path.join(
                        base_data_path, args.dataset, batch_name, "sfm"
                    )
                if args.output_path is None:
                    args.output_path = colmap_model_path = os.path.join(
                        base_data_path, args.dataset, batch_name
                    )
                setup_dataset(
                    batch_name,
                    args.dataset_url,
                    args.download_root,
                    args.delete_archive,
                )

        else:
            batch_name = batches[args.dataset]
            if args.images_path is None:
                args.images_path = os.path.join(
                    base_data_path, batch_name.dataset, args.dataset, "images"
                )
            if args.colmap_model_path is None:
                args.colmap_model_path = os.path.join(
                    base_data_path, batch_name.dataset, args.dataset, "sfm"
                )
            if args.output_path is None:
                args.output_path = colmap_model_path = os.path.join(
                    base_data_path, batch_name.dataset, args.dataset
                )
            setup_dataset(
                args.dataset, args.dataset_url, args.download_root, args.delete_archive
            )

        print("Done!!")

    elif args.action == "split_data":
        print("-" * 75)
        print("Manually splitting the dataset into train and test sets.")
        print("-" * 75)

        train_test_split(
            images_path=args.images_path,
            colmap_model_path=args.colmap_model_path,
            output_path=args.output_path,
            test_size=args.test_size,
            split_strategy=args.split_strategy,
            start_test_data_from=args.start_split_from,
        )
        print("Done!!")

    elif args.action == "create_colmap_bin_files":
        create_colmap_bin_files(args.colmap_model_path)

    elif args.action == "create_sfm_depths":
        print("-" * 75)
        print(f"Creating SfM depth maps")
        print("-" * 75)

        if args.sfm_depths_path is None and args.output_path is None:
            raise ValueError(
                f"Provide a correct path to store the depth maps. {args.sfm_depths_path} was given."
            )
        if args.sfm_depths_path is None:
            args.sfm_depths_path = os.path.join(args.output_path, "depth")
        if not os.path.exists(args.sfm_depths_path):
            os.mkdir(args.sfm_depths_path)

        create_sfm_depth(
            recon_dir=Path(args.colmap_model_path),
            output_dir=Path(args.sfm_depths_path),
            depth_scale_to_integer_factor=1e3,
            min_depth=1e-3,
            max_depth=1e4,
            max_repoj_err=2.5,
            min_n_visible=2,
            include_depth_debug=False,
            input_images_dir=Path(args.images_path),
        )
        print("Done!!")

    elif args.action == "downscale_images":
        print("-" * 75)
        print(f"Downscaling Files (Images and Depth Maps)")
        print("-" * 75)

        downscale_images(
            image_dir=Path(args.images_path),
            num_downscales=args.downscale_factor,
            folder_name=os.path.basename(args.images_path),
        )

        if args.sfm_depths_path is not None:
            downscale_images(
                image_dir=Path(args.sfm_depths_path),
                num_downscales=args.downscale_factor,
                folder_name=os.path.basename(args.sfm_depths_path),
            )
        print("Done!!")
