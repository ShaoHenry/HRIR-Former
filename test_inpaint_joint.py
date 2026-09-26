"""Evaluate a trained joint HRIR inpainting checkpoint.

Supported ``--num_known`` values are 3, 5, 19, and 100. The corresponding
checkpoint is selected automatically. The selected checkpoint path is not
printed during testing.

``--checkpoint`` is kept as an optional explicit override for debugging or for
loading a checkpoint that does not follow the standard naming convention.

When ``--output_dir`` is provided, the script saves:
- one compressed prediction ``.npz`` file per subject;
- one visualization ``.png`` file per subject;
- one ``metrics.json`` file containing the averaged evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import HRTFDataset, collate_fn
from metric import (
    calculate_ild_difference,
    calculate_itd_difference,
    cosine_distance,
    nmse_db,
)
from model_joint import InpaintNet


ALLOWED_NUM_KNOWN = (3, 5, 19, 100)


def resolve_device(requested: str) -> torch.device:
    """Resolve the requested device and fall back to CPU if CUDA is unavailable."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested)


def valid_num_known(value: str) -> int:
    """Argparse validator for the supported known-direction counts."""
    try:
        num_known = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--num_known must be one of: 3, 5, 19, 100"
        ) from exc

    if num_known not in ALLOWED_NUM_KNOWN:
        raise argparse.ArgumentTypeError(
            "--num_known must be one of: 3, 5, 19, 100"
        )

    return num_known


def resolve_checkpoint_path(
    ckpt_dir: str,
    num_known: int,
    checkpoint_override: str | None = None,
) -> Path:
    """Choose the checkpoint silently according to ``num_known``."""
    if num_known not in ALLOWED_NUM_KNOWN:
        raise ValueError(
            "num_known must be one of: 3, 5, 19, 100."
        )

    if checkpoint_override:
        checkpoint_path = Path(checkpoint_override).expanduser()
    else:
        checkpoint_path = (
            Path(ckpt_dir).expanduser()
            / f"checkpoint_{num_known}.pth"
        )

    checkpoint_path = checkpoint_path.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot find checkpoint for num_known={num_known}."
        )

    # Intentionally do not print the selected checkpoint path.
    return checkpoint_path


def load_model(
    checkpoint_path: Path,
    k: int,
    device: torch.device,
) -> InpaintNet:
    """Load a trained InpaintNet checkpoint and switch the model to eval mode."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        # Also support a raw state_dict for convenience.
        state = checkpoint

    model = InpaintNet(K=k)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # Do not print which checkpoint was loaded.
    return model


def sanitize_filename(name: str) -> str:
    """Convert a subject identifier into a safe filename."""
    name = Path(str(name)).name

    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "_")

    return name


def save_prediction_visualization(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    subject: str,
) -> None:
    """Save target, prediction, and absolute-error HRIR heatmaps.

    Parameters
    ----------
    prediction:
        De-normalized predicted HRIRs with shape
        [n_directions, 2 * n_samples].
    target:
        De-normalized ground-truth HRIRs with the same shape.
    mask:
        Direction mask with shape [n_directions].
        A value of 1 indicates a known direction and 0 a missing direction.
    output_path:
        Destination path for the PNG image.
    subject:
        Subject identifier displayed in the figure title.
    """
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Prediction and target must have shape "
            "[n_directions, 2 * n_samples]."
        )

    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target must have identical shapes."
        )

    if prediction.shape[-1] % 2 != 0:
        raise ValueError(
            "The concatenated binaural HRIR length must be divisible by 2."
        )

    n_directions = prediction.shape[0]
    t_ear = prediction.shape[-1] // 2

    target_left = target[:, :t_ear]
    target_right = target[:, t_ear:]

    pred_left = prediction[:, :t_ear]
    pred_right = prediction[:, t_ear:]

    error_left = np.abs(pred_left - target_left)
    error_right = np.abs(pred_right - target_right)

    # Use the same amplitude range for target and prediction.
    amplitude_max = max(
        float(np.max(np.abs(target))),
        float(np.max(np.abs(prediction))),
    )

    if amplitude_max == 0.0:
        amplitude_max = 1.0

    error_max = max(
        float(np.max(error_left)),
        float(np.max(error_right)),
    )

    if error_max == 0.0:
        error_max = 1.0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 8),
        constrained_layout=True,
    )

    image_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "interpolation": "nearest",
        "vmin": -amplitude_max,
        "vmax": amplitude_max,
    }

    error_kwargs = {
        "aspect": "auto",
        "origin": "lower",
        "interpolation": "nearest",
        "vmin": 0.0,
        "vmax": error_max,
    }

    im_target_left = axes[0, 0].imshow(
        target_left,
        **image_kwargs,
    )
    axes[0, 0].set_title("Left Ear - Ground Truth")

    im_pred_left = axes[0, 1].imshow(
        pred_left,
        **image_kwargs,
    )
    axes[0, 1].set_title("Left Ear - Prediction")

    im_error_left = axes[0, 2].imshow(
        error_left,
        **error_kwargs,
    )
    axes[0, 2].set_title("Left Ear - Absolute Error")

    im_target_right = axes[1, 0].imshow(
        target_right,
        **image_kwargs,
    )
    axes[1, 0].set_title("Right Ear - Ground Truth")

    im_pred_right = axes[1, 1].imshow(
        pred_right,
        **image_kwargs,
    )
    axes[1, 1].set_title("Right Ear - Prediction")

    im_error_right = axes[1, 2].imshow(
        error_right,
        **error_kwargs,
    )
    axes[1, 2].set_title("Right Ear - Absolute Error")

    for row in range(2):
        for col in range(3):
            axes[row, col].set_xlabel("Time Sample")
            axes[row, col].set_ylabel("Direction Index")

    # Shared colorbars for the amplitude and error plots.
    fig.colorbar(
        im_pred_left,
        ax=[
            axes[0, 0],
            axes[0, 1],
            axes[1, 0],
            axes[1, 1],
        ],
        label="HRIR Amplitude",
        shrink=0.9,
    )

    fig.colorbar(
        im_error_right,
        ax=[
            axes[0, 2],
            axes[1, 2],
        ],
        label="Absolute Error",
        shrink=0.9,
    )

    num_known = int(np.sum(mask > 0.5))
    num_missing = int(n_directions - num_known)

    fig.suptitle(
        f"Subject: {subject} | "
        f"Known Directions: {num_known} | "
        f"Missing Directions: {num_missing}",
        fontsize=14,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def evaluate(
    args: argparse.Namespace,
) -> dict[str, float | int]:
    """Run evaluation on the validation/test split."""
    device = resolve_device(args.device)

    # Automatically select checkpoint_<num_known>.pth unless --checkpoint is
    # explicitly supplied.
    checkpoint_path = resolve_checkpoint_path(
        ckpt_dir=args.ckpt,
        num_known=args.num_known,
        checkpoint_override=args.checkpoint,
    )

    dataset = HRTFDataset(
        mat_path=args.mat_path,
        split="val",
        mask_ratio=args.mask_ratio,
        subjects_split=args.subjects_split,
        shuffle_subjects=False,
        num_known=args.num_known,
        curriculum=None,
        use_measured=not args.use_simu,
        seed=args.seed,
        split_mode=args.split_mode,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Infer K from the dataset so the test model matches
    # the HRIR dimensionality.
    _, k = dataset[0]["H_norm"].shape

    model = load_model(
        checkpoint_path,
        k,
        device,
    )

    nmse_values: list[float] = []
    cd_values: list[float] = []
    itd_values: list[float] = []
    ild_values: list[float] = []

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else None
    )

    if output_dir is not None:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prediction_dir = output_dir / "predictions"
        visualization_dir = output_dir / "visualizations"

        prediction_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        visualization_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        prediction_dir = None
        visualization_dir = None

    with torch.no_grad():
        for batch in loader:
            h_norm = batch["H_norm"].to(device)
            mask = batch["mask"].to(device)
            geo = batch["geo_feat"].to(device)

            pred = model(
                h_norm,
                mask,
                geo,
                fuse=True,
                return_cues=False,
            ).cpu().numpy()

            gt = batch["H_gt"].numpy()
            masks = batch["mask"].numpy()
            norms = batch["norm"].numpy()

            # Preserve measured/known directions exactly and only use
            # the model prediction for missing directions.
            observed = masks[..., None]

            pred = (
                pred * (1.0 - observed)
                + gt * observed
            )

            for b, subject in enumerate(batch["path"]):
                gt_denorm = gt[b] * norms[b]
                pred_denorm = pred[b] * norms[b]

                # NMSE and cosine distance are evaluated on missing
                # directions according to the metric implementation.
                nmse_values.append(
                    float(
                        nmse_db(
                            gt_denorm,
                            pred_denorm,
                            masks[b],
                        )
                    )
                )

                cd_values.append(
                    float(
                        cosine_distance(
                            gt_denorm,
                            pred_denorm,
                            masks[b],
                        )
                    )
                )

                # Split concatenated binaural HRIR into:
                # [direction, ear, time].
                t_ear = gt_denorm.shape[-1] // 2

                gt_lr = gt_denorm.reshape(
                    gt_denorm.shape[0],
                    2,
                    t_ear,
                )

                pred_lr = pred_denorm.reshape(
                    pred_denorm.shape[0],
                    2,
                    t_ear,
                )

                itd_values.append(
                    float(
                        calculate_itd_difference(
                            gt_lr,
                            pred_lr,
                            fs=args.sample_rate,
                        )
                    )
                )

                ild_values.append(
                    float(
                        calculate_ild_difference(
                            gt_lr,
                            pred_lr,
                        )
                    )
                )

                if output_dir is not None:
                    safe_subject = sanitize_filename(
                        str(subject)
                    )

                    np.savez_compressed(
                        prediction_dir
                        / f"{safe_subject}.npz",
                        prediction=pred_denorm.astype(
                            np.float32
                        ),
                        target=gt_denorm.astype(
                            np.float32
                        ),
                        mask=masks[b].astype(
                            np.float32
                        ),
                    )

                    save_prediction_visualization(
                        prediction=pred_denorm,
                        target=gt_denorm,
                        mask=masks[b],
                        output_path=(
                            visualization_dir
                            / f"{safe_subject}.png"
                        ),
                        subject=safe_subject,
                    )

    metrics: dict[str, float | int] = {
        "num_subjects": len(dataset),
        "num_known": args.num_known,
        "nmse_db": float(
            np.mean(nmse_values)
        ),
        "cosine_distance": float(
            np.mean(cd_values)
        ),
        "itd_error_us": float(
            np.mean(itd_values)
        ),
        "ild_error_db": float(
            np.mean(ild_values)
        ),
    }

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    if output_dir is not None:
        with open(
            output_dir / "metrics.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                metrics,
                f,
                indent=2,
            )

    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--mat_path",
        required=True,
        help="Path to the SONICOM/HRTF MAT file.",
    )

    parser.add_argument(
        "--ckpt",
        default="checkpoints_hrtf",
        help=(
            "Checkpoint directory. The test script loads "
            "<ckpt>/checkpoint_<num_known>.pth automatically."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional explicit checkpoint file. If supplied, "
            "this overrides the automatic --ckpt/--num_known selection."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_known",
        type=valid_num_known,
        default=19,
        help=(
            "Number of known directions. Must be one of: "
            "3, 5, 19, 100. The matching checkpoint is "
            "selected automatically."
        ),
    )

    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--subjects_split",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--split_mode",
        choices=[
            "sonicom",
            "ratio",
        ],
        default="sonicom",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--sample_rate",
        type=int,
        default=48000,
    )

    parser.add_argument(
        "--use_simu",
        action="store_true",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Optional output directory. When specified, "
            "predictions, PNG visualizations, and metrics "
            "are saved."
        ),
    )

    return parser


def main() -> None:
    """Run checkpoint evaluation."""
    args = build_parser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()