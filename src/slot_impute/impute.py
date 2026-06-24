import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from .model import MurmurativeProbe
from .extract import extract_all_weights, extract_slot_pool, inject_all_weights, inject_slot_pool, load_checkpoint
from .variogram import interpolate_krige, interpolate_linear, interpolate_spline, normalize_positions, sample_at_positions


def average_non_slot_weights(
    all_weights: List[Tuple[dict, dict]]
) -> dict:
    keys = ["embed", "q_proj", "k_proj", "v_proj", "output_proj", "lm_head"]
    averaged = {}
    for key in keys:
        weights_list = [w[0][key] for w in all_weights]
        stacked = torch.stack(weights_list)
        averaged[key] = stacked.mean(dim=0)
    averaged["slot_k"] = all_weights[0][0]["slot_k"]
    averaged["slot_v"] = all_weights[0][0]["slot_v"]
    return averaged


def build_imputed_model(
    target_M: int,
    anchor_checkpoints: List[str],
    anchor_Ms: List[int],
    interpolation_method: str = "linear",
    signal_masks: Optional[Dict[int, torch.Tensor]] = None,
    use_signal_only: bool = False,
    non_slot_strategy: str = "average",
) -> Tuple[MurmurativeProbe, dict]:
    all_weights = []
    for path in anchor_checkpoints:
        model, meta = load_checkpoint(path)
        all_weights.append((extract_all_weights(model), meta))

    if non_slot_strategy == "average":
        non_slot = average_non_slot_weights(all_weights)
    else:
        best = min(all_weights, key=lambda x: x[1]["final_ppl"])
        non_slot = best[0]

    del non_slot["slot_k"], non_slot["slot_v"]

    slot_k_list = [w[0]["slot_k"] for w in all_weights]
    slot_v_list = [w[0]["slot_v"] for w in all_weights]

    target_positions = normalize_positions(target_M)

    krige_var_k = None
    krige_var_v = None

    if interpolation_method == "linear":
        slot_k_pred = interpolate_linear(slot_k_list, anchor_Ms, target_M, target_positions)
        slot_v_pred = interpolate_linear(slot_v_list, anchor_Ms, target_M, target_positions)
    elif interpolation_method == "spline":
        slot_k_pred = interpolate_spline(slot_k_list, anchor_Ms, target_M, target_positions)
        slot_v_pred = interpolate_spline(slot_v_list, anchor_Ms, target_M, target_positions)
    elif interpolation_method == "krige":
        slot_k_pred, krige_var_k = interpolate_krige(slot_k_list, anchor_Ms, target_M, target_positions)
        slot_v_pred, krige_var_v = interpolate_krige(slot_v_list, anchor_Ms, target_M, target_positions)

    if signal_masks is not None and use_signal_only:
        target_mask = _interpolate_mask(signal_masks, anchor_Ms, target_M)
        mean_k = torch.stack([sample_at_positions(sk, M, target_positions) for sk, M in zip(slot_k_list, anchor_Ms)]).mean(0)
        mean_v = torch.stack([sample_at_positions(sv, M, target_positions) for sv, M in zip(slot_v_list, anchor_Ms)]).mean(0)
        if target_mask.shape != slot_k_pred.shape:
            target_mask = target_mask.unsqueeze(0).expand(slot_k_pred.shape[0], -1, -1)
        slot_k_pred = torch.where(target_mask, slot_k_pred, mean_k.expand_as(slot_k_pred))
        slot_v_pred = torch.where(target_mask, slot_v_pred, mean_v.expand_as(slot_v_pred))

    model = MurmurativeProbe(num_slots=target_M)
    inject_all_weights(model, non_slot)
    inject_slot_pool(model, slot_k_pred, slot_v_pred)

    return model, {
        "slot_k_pred": slot_k_pred,
        "slot_v_pred": slot_v_pred,
        "krige_var_k": krige_var_k,
        "krige_var_v": krige_var_v,
    }


def _interpolate_mask(
    signal_masks: Dict[int, torch.Tensor],
    anchor_Ms: List[int],
    target_M: int,
) -> torch.Tensor:
    all_Ms = sorted(signal_masks.keys())
    mask_shape = signal_masks[all_Ms[0]].shape
    target_positions = normalize_positions(target_M)

    mask_float_list = []
    for M in all_Ms:
        m = signal_masks[M].float()
        src_pos = normalize_positions(M)
        result = torch.zeros(mask_shape[0], target_M)
        for j in range(target_M):
            p = target_positions[j].item()
            idx = torch.searchsorted(src_pos, p)
            idx = idx.clamp(1, M - 1)
            lo, hi = idx - 1, idx
            alpha = (p - src_pos[lo].item()) / (src_pos[hi].item() - src_pos[lo].item() + 1e-8)
            result[:, j] = (1 - alpha) * m[:, lo] + alpha * m[:, hi]
        mask_float_list.append(result)

    stacked = torch.stack(mask_float_list)
    return stacked.mean(dim=0) > 0.5


def build_imputed_variants(
    anchor_checkpoints: List[str],
    anchor_Ms: List[int],
    target_M: int = 256,
    signal_masks: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Tuple[MurmurativeProbe, dict]]:
    variants = {}

    variants["A_full"] = build_imputed_model(
        target_M, anchor_checkpoints, anchor_Ms,
        interpolation_method="linear",
        non_slot_strategy="average",
    )

    boundary_checkpoints = [cp for cp, M in zip(anchor_checkpoints, anchor_Ms) if M in [128, 512]]
    boundary_Ms = [M for M in anchor_Ms if M in [128, 512]]
    variants["B_boundary"] = build_imputed_model(
        target_M, boundary_checkpoints, boundary_Ms,
        interpolation_method="linear",
        non_slot_strategy="average",
    )

    if signal_masks is not None:
        variants["C_signal"] = build_imputed_model(
            target_M, anchor_checkpoints, anchor_Ms,
            interpolation_method="linear",
            signal_masks=signal_masks,
            use_signal_only=True,
            non_slot_strategy="average",
        )

    variants["D_naive"] = build_imputed_model(
        target_M, boundary_checkpoints, boundary_Ms,
        interpolation_method="linear",
        non_slot_strategy="best",
    )

    return variants


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="experiments/m_variation/config.yaml")
    parser.add_argument("--output-dir", type=str, default="imputed")
    parser.add_argument("--signal-masks-dir", type=str, default="signal_masks")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    unique_anchor_Ms = [m for m in config["anchors"]["m_values"] if m != 256]
    target_M = config["anchors"]["target_m"]
    seeds = config["anchors"]["seeds"]

    os.makedirs(args.output_dir, exist_ok=True)

    anchor_checkpoints = []
    all_anchor_Ms = []
    for M in unique_anchor_Ms:
        for seed in seeds:
            path = os.path.join(args.checkpoint_dir, f"M{M}_seed{seed}.pt")
            if os.path.exists(path):
                anchor_checkpoints.append(path)
                all_anchor_Ms.append(M)

    signal_masks = {}
    if os.path.exists(args.signal_masks_dir):
        for M in unique_anchor_Ms:
            mask_path = os.path.join(args.signal_masks_dir, f"M{M}.pt")
            if os.path.exists(mask_path):
                data = torch.load(mask_path, map_location="cpu", weights_only=False)
                signal_masks[M] = data["signal_mask"]

    variants = build_imputed_variants(
        anchor_checkpoints, all_anchor_Ms, target_M, signal_masks if signal_masks else None
    )

    for name, (model, meta) in variants.items():
        path = os.path.join(args.output_dir, f"{name}.pt")
        MurmurativeProbe.save(model, path, {"variant": name, "target_M": target_M})

    print(f"Saved {len(variants)} imputed variants to {args.output_dir}/")


if __name__ == "__main__":
    _main()