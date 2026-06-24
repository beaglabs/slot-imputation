import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline


def normalize_positions(M: int) -> torch.Tensor:
    if M == 1:
        return torch.tensor([0.0])
    return torch.linspace(0, 1, M)


def sample_at_positions(
    slot_weights: torch.Tensor, M: int, target_positions: torch.Tensor
) -> torch.Tensor:
    src_pos = normalize_positions(M)
    H, _, Dh = slot_weights.shape
    N_t = len(target_positions)

    result = torch.zeros(H, N_t, Dh)
    for i, p in enumerate(target_positions):
        p_val = p.item()
        idx = torch.searchsorted(src_pos, p_val)
        idx = idx.clamp(1, M - 1)
        lo, hi = idx - 1, idx
        alpha = (p_val - src_pos[lo].item()) / (src_pos[hi].item() - src_pos[lo].item() + 1e-8)
        result[:, i] = (1 - alpha) * slot_weights[:, lo] + alpha * slot_weights[:, hi]
    return result


def interpolate_linear(
    anchor_weights: List[torch.Tensor],
    anchor_Ms: List[int],
    target_M: int,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    N_t = len(target_positions)
    H, _, Dh = anchor_weights[0].shape

    sampled = []
    for w, M in zip(anchor_weights, anchor_Ms):
        sampled.append(sample_at_positions(w, M, target_positions))

    stacked = torch.stack(sampled, dim=0)
    Ms_tensor = torch.tensor(anchor_Ms, dtype=torch.float32)

    mean_w = stacked.mean(dim=0)
    Ms_centered = Ms_tensor - Ms_tensor.mean()

    cov = ((stacked - mean_w) * Ms_centered.view(-1, 1, 1, 1)).sum(dim=0)
    var_M = (Ms_centered ** 2).sum()
    slope = cov / (var_M + 1e-8)

    intercept = mean_w - slope * Ms_tensor.mean()
    predicted = slope * float(target_M) + intercept
    return predicted


def interpolate_spline(
    anchor_weights: List[torch.Tensor],
    anchor_Ms: List[int],
    target_M: int,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    N_t = len(target_positions)
    H, _, Dh = anchor_weights[0].shape

    sampled = []
    for w, M in zip(anchor_weights, anchor_Ms):
        sampled.append(sample_at_positions(w, M, target_positions))

    stacked = torch.stack(sampled, dim=0).numpy()
    Ms_arr = np.array(anchor_Ms, dtype=float)

    result = torch.zeros(H, N_t, Dh)
    num_anchors = len(anchor_Ms)

    if num_anchors < 3:
        sorted_idx = np.argsort(Ms_arr)
        return interpolate_linear(anchor_weights, anchor_Ms, target_M, target_positions)

    for h in range(H):
        for p in range(N_t):
            for d in range(Dh):
                vals = stacked[:, h, p, d]
                spline = CubicSpline(Ms_arr, vals)
                result[h, p, d] = float(spline(target_M))

    return result


def interpolate_krige(
    anchor_weights: List[torch.Tensor],
    anchor_Ms: List[int],
    target_M: int,
    target_positions: torch.Tensor,
    sill: Optional[float] = None,
    range_: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N_t = len(target_positions)
    H, _, Dh = anchor_weights[0].shape
    num_anchors = len(anchor_Ms)

    sampled = []
    for w, M in zip(anchor_weights, anchor_Ms):
        sampled.append(sample_at_positions(w, M, target_positions))

    stacked = torch.stack(sampled, dim=0).numpy()
    Ms_arr = np.array(anchor_Ms, dtype=float)

    predicted = torch.zeros(H, N_t, Dh)
    variance = torch.zeros(H, N_t, Dh)

    max_dist = Ms_arr.max() - Ms_arr.min()

    for h in range(H):
        for p in range(N_t):
            for d in range(Dh):
                vals = stacked[:, h, p, d]

                distances = np.abs(Ms_arr[:, None] - Ms_arr[None, :])
                nz = distances > 0
                if nz.sum() > 0:
                    fit_sill = sill if sill is not None else np.maximum(np.var(vals), 1e-6)
                    fit_range = range_ if range_ is not None else max_dist * 0.5
                else:
                    fit_sill = 0.05
                    fit_range = max_dist * 0.5

                gamma_matrix = fit_sill * (1 - np.exp(-distances / (fit_range + 1e-8)))
                gamma_matrix += np.eye(num_anchors) * 1e-10

                gamma_expanded = np.zeros((num_anchors + 1, num_anchors + 1))
                gamma_expanded[:num_anchors, :num_anchors] = gamma_matrix
                gamma_expanded[:num_anchors, -1] = 1.0
                gamma_expanded[-1, :num_anchors] = 1.0

                rhs = np.zeros(num_anchors + 1)
                tgt_dists = np.abs(Ms_arr - float(target_M))
                rhs[:num_anchors] = fit_sill * (1 - np.exp(-tgt_dists / (fit_range + 1e-8)))
                rhs[-1] = 1.0

                try:
                    weights = np.linalg.solve(gamma_expanded, rhs)
                except np.linalg.LinAlgError:
                    weights = np.ones(num_anchors)
                    weights = np.append(weights, 0.0)
                    weights = weights / (weights[:num_anchors].sum() + 1e-8)

                w = weights[:num_anchors]
                lagrange = weights[-1]
                predicted[h, p, d] = float((w * vals).sum())
                variance[h, p, d] = float((w * rhs[:num_anchors]).sum() + lagrange)

    return predicted, variance


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="experiments/m_variation/config.yaml")
    parser.add_argument("--output-dir", type=str, default="variogram_models")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Variogram analysis — nothing to save yet (interpolation is stateless).")
    print("Interpolation functions are ready for use by impute.py.")


if __name__ == "__main__":
    _main()