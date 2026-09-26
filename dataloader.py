"""Dataset utilities for binaural HRIR inpainting."""
from __future__ import annotations

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def collate_fn(batch_list):
    out = {}
    for key in batch_list[0]:
        if key == "path":
            out[key] = [item[key] for item in batch_list]
        else:
            out[key] = torch.stack([item[key] for item in batch_list], dim=0)
    return out


class HRTFDataset(Dataset):
    """One subject per item.

    Expected MAT keys (measured mode): ``HRIR_L_meas``, ``HRIR_R_meas``,
    ``POS_meas``. Simulated mode uses the corresponding ``*_simu`` keys.
    HRIR arrays must have shape ``(n_directions, n_samples, n_subjects)`` and
    positions ``(n_directions, 3, n_subjects)``.
    """

    def __init__(
        self,
        mat_path: str,
        split: str,
        mask_ratio: float,
        num_known: int | None = None,
        mask_minmax=None,
        subjects_split: float = 0.9,
        shuffle_subjects: bool = True,
        curriculum=None,
        use_measured: bool = True,
        seed: int = 42,
        split_mode: str = "sonicom",
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        if split_mode not in {"sonicom", "ratio"}:
            raise ValueError("split_mode must be 'sonicom' or 'ratio'")

        self.num_known = num_known
        self.mask_ratio = float(mask_ratio)
        self.mask_minmax = mask_minmax
        self.curriculum = curriculum
        self._epoch = 0

        data = loadmat(mat_path)
        suffix = "meas" if use_measured else "simu"
        required = [f"HRIR_L_{suffix}", f"HRIR_R_{suffix}", f"POS_{suffix}"]
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"Missing MAT keys: {missing}. Available keys: {sorted(data.keys())}")

        self.HL = np.asarray(data[required[0]], dtype=np.float32)
        self.HR = np.asarray(data[required[1]], dtype=np.float32)
        self.POS = np.asarray(data[required[2]], dtype=np.float32)

        if self.HL.shape != self.HR.shape or self.HL.ndim != 3:
            raise ValueError("Left/right HRIR arrays must be matching 3-D arrays")
        if self.POS.shape != (self.HL.shape[0], 3, self.HL.shape[2]):
            raise ValueError(
                "Position array must have shape (n_directions, 3, n_subjects); "
                f"got {self.POS.shape} for HRIR shape {self.HL.shape}"
            )

        self.n_dirs, self.T, self.n_subj = self.HL.shape
        if num_known is not None and not (1 <= num_known <= self.n_dirs):
            raise ValueError(f"num_known must be in [1, {self.n_dirs}], got {num_known}")

        ids = np.arange(self.n_subj)
        if shuffle_subjects:
            rng = np.random.default_rng(seed)
            rng.shuffle(ids)

        if split_mode == "sonicom":
            # Reproduces the split used by the final SONICOM experiment:
            # shuffled subjects [0:180] for training, with position 78 excluded,
            # and [180:199] for validation.
            if self.n_subj < 199:
                raise ValueError(
                    "split_mode='sonicom' requires at least 199 subjects. "
                    "Use split_mode='ratio' for other datasets."
                )
            if split == "train":
                selected = ids[:180]
                selected = np.delete(selected, 78)
            else:
                selected = ids[180:199]
        else:
            n_train = int(round(self.n_subj * subjects_split))
            n_train = min(max(n_train, 1), self.n_subj - 1)
            selected = ids[:n_train] if split == "train" else ids[n_train:]

        self.subj_ids = selected

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _current_ratio(self):
        if self.curriculum is None:
            return self.mask_ratio
        warmup = int(self.curriculum.get("warmup_epochs", 0))
        start = float(self.curriculum.get("ratio_start", self.mask_ratio))
        final = float(self.curriculum.get("ratio_final", self.mask_ratio))
        t = min(1.0, self._epoch / max(1, warmup))
        return (1.0 - t) * start + t * final

    def _make_mask_ratio(self, n_dirs: int):
        ratio = np.random.uniform(*self.mask_minmax) if self.mask_minmax else self._current_ratio()
        n_missing = max(1, int(round(ratio * n_dirs)))
        idx = np.random.permutation(n_dirs)
        mask = np.ones(n_dirs, dtype=np.float32)
        mask[idx[:n_missing]] = 0.0
        return mask

    def __len__(self):
        return len(self.subj_ids)

    def __getitem__(self, i):
        sid = int(self.subj_ids[i])
        h = np.concatenate([self.HL[:, :, sid], self.HR[:, :, sid]], axis=-1).astype(np.float32)
        geo_feat = self.POS[:, :, sid].astype(np.float32)

        if self.num_known is not None:
            mask = np.zeros(self.n_dirs, dtype=np.float32)
            # Deterministic, uniformly spaced known directions. This reproduces
            # the behavior used by the final --num_known 19 training run.
            mask[np.linspace(0, self.n_dirs - 1, self.num_known, dtype=int)] = 1.0
        else:
            mask = self._make_mask_ratio(self.n_dirs)

        h_t = torch.from_numpy(h)
        mask_t = torch.from_numpy(mask)
        missing = (1.0 - mask_t).unsqueeze(-1)
        norm = torch.abs(h_t * missing).max()
        h_norm = h_t / (norm + 1e-8)

        return {
            "H_norm": h_norm,
            "norm": norm,
            "mask": mask_t,
            "H_gt": h_norm.clone(),
            "geo_feat": torch.from_numpy(geo_feat),
            "path": f"subject_{sid}",
        }
