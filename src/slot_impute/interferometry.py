import json
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_align_slots(
    slot_k_a: torch.Tensor, slot_k_b: torch.Tensor
) -> torch.Tensor:
    H, M, Dh = slot_k_a.shape
    perm = torch.zeros(H, M, dtype=torch.long)
    for h in range(H):
        a = F.normalize(slot_k_a[h], dim=-1)
        b = F.normalize(slot_k_b[h], dim=-1)
        sim = a @ b.T
        cost = -sim.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        perm[h] = torch.tensor(col_ind)
    return perm


def align_all_seeds(checkpoints: dict) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    from .extract import load_checkpoint, extract_slot_pool

    seeds = sorted(checkpoints.keys())
    if not seeds:
        raise ValueError("No checkpoints provided to align_all_seeds")

    ref_seed = seeds[0]
    ref_model, _ = load_checkpoint(checkpoints[ref_seed])
    ref_k, ref_v = extract_slot_pool(ref_model)

    aligned_k = [ref_k]
    aligned_v = [ref_v]

    for seed in seeds[1:]:
        model, _ = load_checkpoint(checkpoints[seed])
        sk, sv = extract_slot_pool(model)
        perm = hungarian_align_slots(ref_k, sk)
        H = sk.shape[0]
        sk_aligned = sk.clone()
        sv_aligned = sv.clone()
        for h in range(H):
            sk_aligned[h] = sk[h, perm[h]]
            sv_aligned[h] = sv[h, perm[h]]
        aligned_k.append(sk_aligned)
        aligned_v.append(sv_aligned)

    return aligned_k, aligned_v


def compute_signal_mask(
    aligned_slot_k: List[torch.Tensor],
    aligned_slot_v: List[torch.Tensor],
    threshold="median",
) -> Tuple[torch.Tensor, torch.Tensor]:
    stacked_k = torch.stack(aligned_slot_k)
    stacked_v = torch.stack(aligned_slot_v)

    var_k = stacked_k.var(dim=0)
    var_v = stacked_v.var(dim=0)
    variance = var_k + var_v

    if threshold == "median":
        thresh = variance.median()
    elif isinstance(threshold, (int, float)):
        thresh = float(threshold)
    else:
        raise ValueError(f"Unknown threshold: {threshold}")

    signal_mask = variance < thresh
    return signal_mask, variance


def interferometry_report(alignments_by_M: dict) -> dict:
    stats = {}
    for M, (aligned_k, aligned_v) in alignments_by_M.items():
        signal_mask, variance = compute_signal_mask(aligned_k, aligned_v)
        total = signal_mask.numel()
        signal_count = signal_mask.sum().item()
        stats[M] = {
            "total_weights": total,
            "signal_weights": signal_count,
            "signal_ratio": signal_count / total,
            "mean_variance": variance.mean().item(),
            "median_variance": variance.median().item(),
        }
    return stats


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="experiments/m_variation/config.yaml")
    parser.add_argument("--output-dir", type=str, default="signal_masks")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    m_values = config["anchors"]["m_values"]
    seeds = config["anchors"]["seeds"]

    os.makedirs(args.output_dir, exist_ok=True)

    alignments_by_M = {}
    for M in m_values:
        checkpoints = {}
        missing = []
        for seed in seeds:
            path = os.path.join(args.checkpoint_dir, f"M{M}_seed{seed}.pt")
            if os.path.exists(path):
                checkpoints[seed] = path
            else:
                missing.append(seed)
        if missing:
            print(f"Skipping M={M}: missing seeds {missing}")
            continue
        if len(checkpoints) < 2:
            print(f"Skipping M={M}: need at least 2 seeds, found {len(checkpoints)}")
            continue
        aligned_k, aligned_v = align_all_seeds(checkpoints)
        alignments_by_M[M] = (aligned_k, aligned_v)

        mask, var = compute_signal_mask(aligned_k, aligned_v)
        torch.save({"signal_mask": mask, "variance": var}, os.path.join(args.output_dir, f"M{M}.pt"))

    stats = interferometry_report(alignments_by_M)
    with open(os.path.join(args.output_dir, "interferometry_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _main()