"""Core checkpoint, cue, and visualization utilities."""
from __future__ import annotations

import os
import numpy as np
import torch
import matplotlib.pyplot as plt


def ild_db_from_hrir(H, num_bands=12, eps=1e-8):
    """Band-wise ILD targets from concatenated [left, right] HRIR tensors."""
    _, _, k = H.shape
    t = k // 2
    left, right = H[..., :t], H[..., t:]
    xl, xr = torch.fft.rfft(left, dim=-1), torch.fft.rfft(right, dim=-1)
    el = xl.real ** 2 + xl.imag ** 2
    er = xr.real ** 2 + xr.imag ** 2
    edges = torch.linspace(0, el.shape[-1], num_bands + 1, device=H.device).long()
    ild = []
    for band in range(num_bands):
        lo, hi = edges[band], edges[band + 1]
        e_left = el[..., lo:hi].sum(-1)
        e_right = er[..., lo:hi].sum(-1)
        ild.append(10 * torch.log10((e_left + eps) / (e_right + eps)))
    return torch.stack(ild, dim=-1)


def itd_from_hrir(H, max_lag=32):
    """Integer-sample ITD target from cross-correlation."""
    _, _, k = H.shape
    t = k // 2
    left, right = H[..., :t], H[..., t:]
    lags = torch.arange(-max_lag, max_lag + 1, device=H.device)
    corrs = []
    for lag_tensor in lags:
        lag = int(lag_tensor.item())
        if lag >= 0:
            corr = (left[..., : t - lag] * right[..., lag:]).sum(-1)
        else:
            offset = -lag
            corr = (left[..., offset:] * right[..., : t - offset]).sum(-1)
        corrs.append(corr)
    idx = torch.stack(corrs, dim=-1).argmax(-1)
    return lags[idx].float()


def load_checkpoint_if_available(model, optimizer, checkpoint_path, map_location="cpu"):
    if not os.path.exists(checkpoint_path):
        return 0
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    state = checkpoint.get("optimizer_state_dict")
    if optimizer is not None and state is not None:
        try:
            optimizer.load_state_dict(state)
        except Exception as exc:
            print(f"[WARN] Could not load optimizer state: {exc}")
    return int(checkpoint.get("epoch", 0))


def save_checkpoint(model, optimizer, epoch, out_path):
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        },
        out_path,
    )


def _normalize_image(array2d, eps=1e-8):
    vmin, vmax = float(array2d.min()), float(array2d.max())
    return (array2d - vmin) / (vmax - vmin + eps)


def visualize_inpainting_hrir(H_gt, H_pred, mask, save_dir, max_samples=4, T_ear=256):
    """Save target/masked/prediction/error maps for a few validation subjects."""
    os.makedirs(save_dir, exist_ok=True)
    for b in range(min(H_gt.shape[0], max_samples)):
        for ear_name, sl in (("L", slice(0, T_ear)), ("R", slice(T_ear, 2 * T_ear))):
            gt = H_gt[b, :, sl]
            pred = H_pred[b, :, sl]
            masked = gt.copy()
            masked[mask[b] == 0] = 0.0
            for suffix, image in (
                ("gt", gt),
                ("masked", masked),
                ("pred", pred),
                ("err", np.abs(gt - pred)),
            ):
                plt.imsave(
                    os.path.join(save_dir, f"sample{b}_{ear_name}_{suffix}.png"),
                    _normalize_image(image),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
