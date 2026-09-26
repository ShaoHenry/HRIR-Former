"""Evaluation metrics for HRIR inpainting."""
from __future__ import annotations

import math
import numpy as np
import scipy.signal as signal


def itd_estimator_maxiacce(hrir, fs, upper_cut_freq=3000, filter_order=10):
    """Estimate ITD using low-pass filtered envelope cross-correlation."""
    itd_samples = []
    maxiacc = []
    b, a = signal.butter(filter_order, upper_cut_freq / (fs / 2))
    for loc in hrir:
        left = signal.lfilter(b, a, loc[0])
        right = signal.lfilter(b, a, loc[1])
        corr = signal.correlate(np.abs(signal.hilbert(left)), np.abs(signal.hilbert(right)))
        maxiacc.append(np.max(np.abs(corr)))
        idx_lag = np.argmax(np.abs(corr))
        itd_samples.append(idx_lag - np.shape(hrir)[2])
    return np.asarray(itd_samples) / fs, itd_samples, maxiacc


def calculate_itd_difference(hrir1, hrir2, fs=48000):
    itd1, _, _ = itd_estimator_maxiacce(hrir1, fs)
    itd2, _, _ = itd_estimator_maxiacce(hrir2, fs)
    return float(np.mean(np.abs(itd1 - itd2)) * 1e6)


def _broadband_ild_db(hrir, eps=1e-12):
    left_rms = np.sqrt(np.mean(np.square(hrir[:, 0, :]), axis=-1) + eps)
    right_rms = np.sqrt(np.mean(np.square(hrir[:, 1, :]), axis=-1) + eps)
    return 20.0 * np.log10((left_rms + eps) / (right_rms + eps))


def calculate_ild_difference(hrir1, hrir2, average=True):
    diff = _broadband_ild_db(hrir1) - _broadband_ild_db(hrir2)
    return float(np.mean(np.abs(diff))) if average else diff


def nmse_db(H_true: np.ndarray, H_hat: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask == 0)[0]
    if len(idx) == 0:
        return 0.0
    diff = H_hat[idx] - H_true[idx]
    num = float(np.sum(diff ** 2))
    den = float(np.sum(H_true[idx] ** 2)) + 1e-12
    return 10.0 * math.log10(num / den + 1e-12)


def cosine_distance(H_true: np.ndarray, H_hat: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask == 0)[0]
    if len(idx) == 0:
        return 0.0
    vals = []
    for i in idx:
        a, b = H_true[i], H_hat[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        vals.append(1.0 - cos ** 2)
    return float(np.mean(vals))
