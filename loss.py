"""Training loss for joint HRIR reconstruction and binaural cues."""
import torch
import torch.nn.functional as F

from utils import ild_db_from_hrir, itd_from_hrir


def hrir_cue_loss(
    H_pred,
    H_gt,
    mask,
    itd_pred,
    ild_pred_db,
    num_ild_bands=12,
    lambda_itd=0.05,
    lambda_ild=0.05,
):
    missing = (mask == 0).unsqueeze(-1).float()
    hrir_loss = (((H_pred - H_gt) ** 2 * missing).sum() / (missing.sum() + 1e-8))

    with torch.no_grad():
        itd_gt = itd_from_hrir(H_gt)
        ild_gt = ild_db_from_hrir(H_gt, num_bands=num_ild_bands)

    itd_loss = F.smooth_l1_loss(itd_pred, itd_gt)
    ild_loss = F.l1_loss(ild_pred_db, ild_gt)
    total = hrir_loss + lambda_itd * itd_loss + lambda_ild * ild_loss
    return total, {
        "hrir": hrir_loss.item(),
        "itd": itd_loss.item(),
        "ild": ild_loss.item(),
    }
