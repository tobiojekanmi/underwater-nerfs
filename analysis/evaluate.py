import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import pandas as pd
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import pycolmap
from numpy.typing import NDArray

from .uciqe import evaluate_restored

psnr = PeakSignalNoiseRatio(data_range=1.0)
ssim = structural_similarity_index_measure
lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)



def evaluate_model(
    model_path,
    gt_path,
    render_path,
    output_dir_path,
    output_filename,
    restored_path=None,
    overwrite=False,
    evaluate_only_water_metrics=False,
):
    """
    Evaluate the images using SSIM, PSNR, LPIPS, UCIQE, and UIQM.

    Args:
        model_path (str): Path to the model.
        gt_path (str): Path to the ground truth images.
        render_path (str): Path to the rendered images.
        output_path (str): Path to save the results.
        output_filename (str): Name of the output file.
        restored_path (str): Path to save the restored images.
        overwrite (bool): Overwrite the output file if it exists.
        evaluate_only_water_metrics (bool): Evaluate only UCIQE and UIQM.
    """
    output_filename = (
        output_filename + ".csv" if ".csv" not in output_filename else output_filename
    )
    output_path = os.path.join(output_dir_path, output_filename)

    if os.path.exists(output_path):
        print("Output file already exists.")
        return

    # Create output path if not exists
    os.makedirs(output_dir_path, exist_ok=True)

    # Read the files
    filenames = os.listdir(gt_path)

    results = []

    reconstruction = pycolmap.Reconstruction(model_path)

    for file in tqdm(filenames):
        if not evaluate_only_water_metrics:
            # Read the images
            gt_image_path = os.path.join(gt_path, file)
            rend_image_path = os.path.join(render_path, file)

            gt_image = np.array(Image.open(gt_image_path))
            rend_image = np.array(Image.open(rend_image_path))

            # Apply transformations
            gt_image_tensor = torch.from_numpy(gt_image) / 255.0
            rend_image_tensor = torch.from_numpy(rend_image) / 255.0
            gt_image_tensor = torch.moveaxis(gt_image_tensor, -1, 0)[None, ...]
            rend_image_tensor = torch.moveaxis(rend_image_tensor, -1, 0)[None, ...]

            # Calculate SSIM and PSNR
            ssim_value = ssim(gt_image_tensor, rend_image_tensor)
            psnr_value = psnr(gt_image_tensor, rend_image_tensor)
            lpips_value = lpips(gt_image_tensor, rend_image_tensor)

            for _, image in reconstruction.images.items():
                if file[:-4] in image.name:
                    Q = image.cam_from_world.rotation.quat
                    T = image.cam_from_world.translation.reshape(3, 1)
                    R = quaternion_to_matrix(torch.from_numpy(Q))
                    CC = -R.T @ torch.from_numpy(T)

            result = {
                "Filename": file,
                "CCX": round(CC[0][0].item(), 2),
                "CCY": round(CC[1][0].item(), 2),
                "CCZ": round(CC[2][0].item(), 2),
                "SSIM": round(ssim_value.item(), 2),
                "PSNR": round(psnr_value.item(), 2),
                "LPIPS": round(lpips_value.item(), 2),
            }

        else :
            result = {
                "Filename": file,
            }

        if restored_path is not None:
            restored_image_path = os.path.join(restored_path, file)
            restored_image = np.array(Image.open(restored_image_path))
            uiqm_value, uciqe_value = evaluate_restored(restored_image)
            result["UIQM"] = round(uiqm_value, 2)
            result["UCIQE"] = round(uciqe_value, 2)

        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)


def get_metrics_means(train_df, test_df):
    output = {
        "train_psnr": np.round(train_df["PSNR"].mean(), 2),
        "train_ssim": np.round(train_df["SSIM"].mean(), 2),
        "train_lpips": np.round(train_df["LPIPS"].mean(), 2),
        "test_psnr": np.round(test_df["PSNR"].mean(), 2),
        "test_ssim": np.round(test_df["SSIM"].mean(), 2),
        "test_lpips": np.round(test_df["LPIPS"].mean(), 2),
    }

    if "UIQM" in train_df.columns and "UCIQE" in train_df.columns:
        output["train_uiqm"] = np.round(train_df["UIQM"].mean(), 2)
        output["train_uciqe"] = np.round(train_df["UCIQE"].mean(), 2)
    if "UIQM" in test_df.columns and "UCIQE" in test_df.columns:
        output["test_uiqm"] = np.round(test_df["UIQM"].mean(), 2)
        output["test_uciqe"] = np.round(test_df["UCIQE"].mean(), 2)

    return output


def quaternion_to_matrix(quaternion: NDArray) -> np.ndarray:
    """
    Returns a rotation matrix from quaternion.

    Args:
        quaternion: value to convert to matrix
    """

    _EPS = np.finfo(float).eps * 4.0
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array(
        [
            [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
            [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
            [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
        ]
    )