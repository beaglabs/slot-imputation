import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from .transformer import MiniGPT2, TransformerConfig


def collect_mlp_activations(
    model: MiniGPT2,
    batches: List[dict],
    num_batches: int = 10,
) -> Dict[int, torch.Tensor]:
    activations: Dict[int, List[torch.Tensor]] = {}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(module, input):
            if layer_idx not in activations:
                activations[layer_idx] = []
            activations[layer_idx].append(input[0].detach())

        return hook

    for i, block in enumerate(model.blocks):
        if block.mlp is not None:
            hooks.append(
                block.mlp.W2.register_forward_pre_hook(make_hook(i))
            )

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for idx in range(min(num_batches, len(batches))):
            input_ids = batches[idx]["input_ids"].to(device)
            model(input_ids)

    for h in hooks:
        h.remove()

    result = {}
    for layer_idx, act_list in activations.items():
        all_acts = torch.cat(act_list, dim=0)
        all_acts_flat = all_acts.view(-1, all_acts.shape[-1])
        result[layer_idx] = all_acts_flat.mean(dim=0)

    return result


def upsample_weight_matrix(
    src_matrix: torch.Tensor,
    dim_to_upsample: int,
    src_act: torch.Tensor,
    target_dim: int,
) -> torch.Tensor:
    if dim_to_upsample == 0:
        src = src_matrix
    else:
        src = src_matrix

    sort_idx = torch.argsort(src_act, descending=True)

    src_pos = torch.linspace(0, 1, len(src_act))
    tgt_pos = torch.linspace(0, 1, target_dim)

    if dim_to_upsample == 0:
        sorted_src = src[sort_idx]
        result = torch.zeros(target_dim, src.shape[1], device=src.device, dtype=src.dtype)
        for j, p in enumerate(tgt_pos):
            idx = int(torch.searchsorted(src_pos, p).clamp(1, len(src_act) - 1).item())
            lo, hi = idx - 1, idx
            alpha = (p - src_pos[lo].item()) / (src_pos[hi].item() - src_pos[lo].item() + 1e-8)
            result[j] = (1 - alpha) * sorted_src[lo] + alpha * sorted_src[hi]
    else:
        sorted_src = src[:, sort_idx]
        result = torch.zeros(src.shape[0], target_dim, device=src.device, dtype=src.dtype)
        for j, p in enumerate(tgt_pos):
            idx = int(torch.searchsorted(src_pos, p).clamp(1, len(src_act) - 1).item())
            lo, hi = idx - 1, idx
            alpha = (p - src_pos[lo].item()) / (src_pos[hi].item() - src_pos[lo].item() + 1e-8)
            result[:, j] = (1 - alpha) * sorted_src[:, lo] + alpha * sorted_src[:, hi]

    return result


def impute_mlp_to_target(
    source_model: MiniGPT2,
    target_d_ff: int,
    batches: List[dict],
    num_act_batches: int = 10,
) -> MiniGPT2:
    device = next(source_model.parameters()).device
    source_config = source_model.config

    activations = collect_mlp_activations(source_model, batches, num_act_batches)

    target_config = TransformerConfig(
        vocab_size=source_config.vocab_size,
        d_model=source_config.d_model,
        d_ff=target_d_ff,
        n_heads=source_config.n_heads,
        n_layers=source_config.n_layers,
        max_seq_len=source_config.max_seq_len,
        use_mlp=source_config.use_mlp,
    )
    target_model = MiniGPT2(target_config).to(device)

    source_sd = source_model.state_dict()
    target_sd = target_model.state_dict()

    for key in target_sd:
        if key in source_sd and target_sd[key].shape == source_sd[key].shape:
            target_sd[key].copy_(source_sd[key])

    for layer_idx in range(source_config.n_layers):
        w1_key = f"blocks.{layer_idx}.mlp.W1.weight"
        w2_key = f"blocks.{layer_idx}.mlp.W2.weight"

        if w1_key not in source_sd:
            continue

        w1_src = source_sd[w1_key]
        w2_src = source_sd[w2_key]
        act = activations[layer_idx].to(device)

        w1_up = upsample_weight_matrix(w1_src, dim_to_upsample=0, src_act=act, target_dim=target_d_ff)
        w2_up = upsample_weight_matrix(w2_src, dim_to_upsample=1, src_act=act, target_dim=target_d_ff)

        target_sd[w1_key].copy_(w1_up)
        target_sd[w2_key].copy_(w2_up)

    target_model.load_state_dict(target_sd)
    return target_model


def compute_param_counts(
    d_model: int,
    d_ff: int,
    n_layers: int,
    vocab_size: int,
    n_heads: int,
) -> dict:
    per_layer_attn = 4 * d_model * d_model
    per_layer_mlp = 2 * d_model * d_ff
    per_layer_ln = 2 * d_model

    embed_params = vocab_size * d_model
    pos_embed_params = 1024 * d_model

    attn_params = n_layers * per_layer_attn
    mlp_params = n_layers * per_layer_mlp
    ln_params = n_layers * per_layer_ln * 2

    total = embed_params + pos_embed_params + attn_params + mlp_params + ln_params

    return {
        "d_ff": d_ff,
        "total_params": total,
        "mlp_params": mlp_params,
        "non_mlp_params": total - mlp_params,
    }