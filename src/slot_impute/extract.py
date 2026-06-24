from typing import Tuple

import torch
import torch.nn as nn


def load_checkpoint(path: str) -> Tuple[nn.Module, dict]:
    from .model import MurmurativeProbe

    return MurmurativeProbe.load(path)


def extract_slot_pool(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        model.slot_k_emb.data.detach().clone(),
        model.slot_v_emb.data.detach().clone(),
    )


def extract_all_weights(model: nn.Module) -> dict:
    return {
        "embed": model.embed.weight.data.detach().clone(),
        "q_proj": model.q_proj.weight.data.detach().clone(),
        "k_proj": model.k_proj.weight.data.detach().clone(),
        "v_proj": model.v_proj.weight.data.detach().clone(),
        "slot_k": model.slot_k_emb.data.detach().clone(),
        "slot_v": model.slot_v_emb.data.detach().clone(),
        "output_proj": model.output_proj.weight.data.detach().clone(),
        "lm_head": model.lm_head.weight.data.detach().clone(),
    }


def inject_slot_pool(
    model: nn.Module, slot_k: torch.Tensor, slot_v: torch.Tensor
) -> nn.Module:
    model.slot_k_emb.data.copy_(slot_k)
    model.slot_v_emb.data.copy_(slot_v)
    return model


def inject_all_weights(model: nn.Module, weights_dict: dict) -> nn.Module:
    model.embed.weight.data.copy_(weights_dict["embed"])
    model.q_proj.weight.data.copy_(weights_dict["q_proj"])
    model.k_proj.weight.data.copy_(weights_dict["k_proj"])
    model.v_proj.weight.data.copy_(weights_dict["v_proj"])
    model.output_proj.weight.data.copy_(weights_dict["output_proj"])
    model.lm_head.weight.data.copy_(weights_dict["lm_head"])
    if "slot_k" in weights_dict:
        model.slot_k_emb.data.copy_(weights_dict["slot_k"])
    if "slot_v" in weights_dict:
        model.slot_v_emb.data.copy_(weights_dict["slot_v"])
    return model