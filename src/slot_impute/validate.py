import json
import math
import os
from typing import Dict, Iterator, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from .model import MurmurativeProbe, build_synthetic_data
from .extract import extract_slot_pool, load_checkpoint
from .interferometry import hungarian_align_slots


def compute_perplexity(
    model: nn.Module,
    data_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    device: str = "cpu",
    num_batches: int = 10,
) -> float:
    model.eval()
    model.to(device)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for input_ids, target_ids in data_iter:
            logits = model(input_ids.to(device))
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                target_ids.view(-1).to(device),
            )
            total_loss += loss.item()
            count += 1
            if count >= num_batches:
                break
    avg_loss = total_loss / max(count, 1)
    return math.exp(min(avg_loss, 20))


def weight_distance(
    imputed_model: nn.Module, ground_truth_model: nn.Module
) -> dict:
    gt_k, gt_v = extract_slot_pool(ground_truth_model)
    im_k, im_v = extract_slot_pool(imputed_model)

    perm = hungarian_align_slots(gt_k, im_k)
    H = im_k.shape[0]
    im_k_aligned = im_k.clone()
    im_v_aligned = im_v.clone()
    for h in range(H):
        im_k_aligned[h] = im_k[h, perm[h]]
        im_v_aligned[h] = im_v[h, perm[h]]

    slot_k_mse = F.mse_loss(im_k_aligned, gt_k).item()
    slot_v_mse = F.mse_loss(im_v_aligned, gt_v).item()

    k_flat = im_k_aligned.reshape(H, -1)
    v_flat = im_v_aligned.reshape(H, -1)
    gt_k_flat = gt_k.reshape(H, -1)
    gt_v_flat = gt_v.reshape(H, -1)

    cos_k = F.cosine_similarity(k_flat, gt_k_flat, dim=-1).mean().item()
    cos_v = F.cosine_similarity(v_flat, gt_v_flat, dim=-1).mean().item()

    return {
        "slot_k_mse": slot_k_mse,
        "slot_v_mse": slot_v_mse,
        "slot_k_cosine": cos_k,
        "slot_v_cosine": cos_v,
    }


def convergence_speedup(
    imputed_model: nn.Module,
    random_init_model: nn.Module,
    num_steps: int = 100,
    device: str = "cpu",
) -> dict:
    imputed_model.to(device)
    random_init_model.to(device)

    imputed_losses = _fine_tune(imputed_model, num_steps, device)
    random_losses = _fine_tune(random_init_model, num_steps, device)

    imputed_initial = imputed_losses[0]
    random_initial = random_losses[0]
    speedup_steps = num_steps
    for i, loss in enumerate(random_losses):
        if loss <= imputed_initial:
            speedup_steps = i + 1
            break

    speedup_ratio = speedup_steps

    return {
        "imputed_loss_curve": imputed_losses,
        "random_loss_curve": random_losses,
        "imputed_initial_loss": imputed_initial,
        "random_initial_loss": random_initial,
        "speedup_ratio": speedup_ratio,
    }


def _fine_tune(model: nn.Module, num_steps: int, device: str) -> list[float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-4)
    losses = []

    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    vocab_size = 256
    seq_len = 128

    for step in range(num_steps):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=torch.long, generator=rng)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), input_ids.view(-1)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())

    return losses


def calibration_analysis(
    krige_variance: torch.Tensor, imputation_error: torch.Tensor
) -> dict:
    var_flat = krige_variance.flatten().numpy()
    err_flat = imputation_error.flatten().numpy()

    if len(var_flat) < 10:
        return {"spearman_rho": 0.0, "decile_errors": []}

    rho, _ = spearmanr(var_flat, err_flat)

    sorted_idx = np.argsort(var_flat)
    decile_errors = []
    n = len(sorted_idx)
    for d in range(10):
        start = d * n // 10
        end = (d + 1) * n // 10
        if end > start:
            decile_errors.append(float(np.abs(err_flat[sorted_idx[start:end]]).mean()))

    return {"spearman_rho": float(rho), "decile_errors": decile_errors}


def full_validation_report(
    variants: Dict[str, Tuple[nn.Module, dict]],
    ground_truth_model: nn.Module,
    device: str = "cpu",
) -> dict:
    report = {}
    for name, (model, meta) in variants.items():
        data_iter = build_synthetic_data(256, 128, 10, device, dtype=torch.long)
        entry = {
            "zero_shot_ppl": compute_perplexity(model, data_iter, device),
            "weight_distance": weight_distance(model, ground_truth_model),
            "convergence": convergence_speedup(
                model,
                MurmurativeProbe(num_slots=model.num_slots),
                device=device,
            ),
        }

        if meta.get("krige_var_k") is not None:
            gt_k, _ = extract_slot_pool(ground_truth_model)
            error_k = (meta["slot_k_pred"] - gt_k).abs()
            entry["calibration"] = calibration_analysis(meta["krige_var_k"], error_k)

        report[name] = entry

    return report


import numpy as np


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--imputed-dir", type=str, default="imputed")
    parser.add_argument("--config", type=str, default="experiments/m_variation/config.yaml")
    parser.add_argument("--output", type=str, default="validation_results.json")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    target_M = config["anchors"]["target_m"]
    gt_path = os.path.join(args.checkpoint_dir, f"M{target_M}_seed42.pt")
    gt_model, _ = load_checkpoint(gt_path)

    variant_names = ["A_full", "B_boundary", "C_signal", "D_naive", "E_krige"]
    variants = {}
    for name in variant_names:
        path = os.path.join(args.imputed_dir, f"{name}.pt")
        if os.path.exists(path):
            model, meta = load_checkpoint(path)
            variants[name] = (model, meta)

    validation = full_validation_report(variants, gt_model, device=config.get("validation", {}).get("device", "cpu"))

    with open(args.output, "w") as f:
        json.dump(validation, f, indent=2)

    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    _main()