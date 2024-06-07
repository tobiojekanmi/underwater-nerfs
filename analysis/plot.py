import numpy as np
import matplotlib.pyplot as plt


def plot_nerf_metrics(train_df, test_df, metadata):
    """
    Plot histograms of the performance evaluation metrics
    PSNR, SSIM, and LPIPS for the train and test sets.
    """

    train_psnr = train_df["PSNR"].to_numpy()
    train_ssim = train_df["SSIM"].to_numpy()
    train_lpips = train_df["LPIPS"].to_numpy()

    test_psnr = test_df["PSNR"].to_numpy()
    test_ssim = test_df["SSIM"].to_numpy()
    test_lpips = test_df["LPIPS"].to_numpy()

    fig, ax = plt.subplots(1, 3, figsize=(26, 5))

    # Normalize histograms
    train_psnr_density, train_psnr_bins, _ = ax[0].hist(
        train_psnr,
        bins=20,
        color="blue",
        alpha=1.0,
        label="Train",
        density=True,
    )
    test_psnr_density, test_psnr_bins, _ = ax[0].hist(
        test_psnr,
        bins=20,
        color="red",
        alpha=0.5,
        label="Test",
        density=True,
    )
    ax[0].axvline(
        train_psnr.mean(),
        color="black",
        linestyle="dashed",
        linewidth=2,
        label=f"Train Mean: {train_psnr.mean():.2f}",
    )
    ax[0].axvline(
        test_psnr.mean(),
        color="cyan",
        linestyle="dashed",
        linewidth=2,
        label=f"Test Mean: {test_psnr.mean():.2f}",
    )
    ax[0].set_xlim(5, 40)
    ax[0].set_xlabel("PSNR")
    ax[0].set_ylabel("Probability Density")
    ax[0].legend()

    train_ssim_density, train_ssim_bins, _ = ax[1].hist(
        train_ssim,
        bins=20,
        color="blue",
        alpha=1.0,
        label="Train",
        density=True,
    )
    test_ssim_density, test_ssim_bins, _ = ax[1].hist(
        test_ssim,
        bins=20,
        color="red",
        alpha=0.5,
        label="Test",
        density=True,
    )
    ax[1].axvline(
        train_ssim.mean(),
        color="black",
        linestyle="dashed",
        linewidth=2,
        label=f"Train Mean: {train_ssim.mean():.2f}",
    )
    ax[1].axvline(
        test_ssim.mean(),
        color="cyan",
        linestyle="dashed",
        linewidth=2,
        label=f"Test Mean: {test_ssim.mean():.2f}",
    )
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("SSIM")
    ax[1].set_ylabel("Probability Density")
    ax[1].legend()

    train_lpips_density, train_lpips_bins, _ = ax[2].hist(
        train_lpips,
        bins=20,
        color="blue",
        alpha=1.0,
        label="Train",
        density=True,
    )
    test_lpips_density, test_lpips_bins, _ = ax[2].hist(
        test_lpips,
        bins=20,
        color="red",
        alpha=0.5,
        label="Test",
        density=True,
    )
    ax[2].axvline(
        train_lpips.mean(),
        color="black",
        linestyle="dashed",
        linewidth=2,
        label=f"Train Mean: {train_lpips.mean():.2f}",
    )
    ax[2].axvline(
        test_lpips.mean(),
        color="cyan",
        linestyle="dashed",
        linewidth=2,
        label=f"Test Mean: {test_lpips.mean():.2f}",
    )
    ax[2].set_xlim(0, 1)
    ax[2].set_xlabel("LPIPS")
    ax[2].set_ylabel("Probability Density")
    ax[2].legend()

    if "save_path" in metadata.keys() and metadata["save_path"] is not None:
        fig.savefig(metadata["save_path"])

    plt.show()



def plot_retoration_metrics(train_df, test_df, metadata):
    """
    Plot histograms of the performance evaluation metrics
    PSNR, SSIM, and LPIPS for the train and test sets.
    """

    train_uiqm = train_df["UIQM"].to_numpy()
    train_uciqe = train_df["UCIQE"].to_numpy()
    test_uiqm = test_df["UIQM"].to_numpy()
    test_uciqe = test_df["UCIQE"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(18, 5))

    ax[0].hist(
        train_uciqe,
        bins=20,
        color="blue",
        alpha=1.0,
        label="Train",
        density=True,
    )
    ax[0].hist(
        test_uciqe,
        bins=20,
        color="red",
        alpha=0.5,
        label="Test",
        density=True,
    )
    ax[0].axvline(
        train_uciqe.mean(),
        color="black",
        linestyle="dashed",
        linewidth=2,
        label=f"Train Mean: {train_uciqe.mean():.2f}",
    )
    ax[0].axvline(
        test_uciqe.mean(),
        color="cyan",
        linestyle="dashed",
        linewidth=2,
        label=f"Test Mean: {test_uciqe.mean():.2f}",
    )
    ax[0].set_xlim(0, 35)
    ax[0].set_xlabel("UCIQE")
    ax[0].set_ylabel("Probability Density")
    ax[0].legend()

    ax[1].hist(
        train_uiqm,
        bins=20,
        color="blue",
        alpha=1.0,
        label="Train",
        density=True,
    )
    ax[1].hist(
        test_uiqm,
        bins=20,
        color="red",
        alpha=0.5,
        label="Test",
        density=True,
    )
    ax[1].axvline(
        train_uiqm.mean(),
        color="black",
        linestyle="dashed",
        linewidth=2,
        label=f"Train Mean: {train_uiqm.mean():.2f}",
    )
    ax[1].axvline(
        test_uiqm.mean(),
        color="cyan",
        linestyle="dashed",
        linewidth=2,
        label=f"Test Mean: {test_uiqm.mean():.2f}",
    )
    ax[1].set_xlim(-1, 2.0)
    ax[1].set_xlabel("UIQM")
    ax[1].set_ylabel("Probability Density")
    ax[1].legend()

    if "save_path" in metadata.keys() and metadata["save_path"] is not None:
        fig.savefig(metadata["save_path"])

    plt.show()

