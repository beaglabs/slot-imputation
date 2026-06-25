import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .data import load_wikitext_batches, wikitext_batch_iterator
from .impute_mlp import impute_mlp_to_target, compute_param_counts
from .transformer import MiniGPT2, TransformerConfig


@torch.no_grad()
def compute_ppl(
    model: MiniGPT2,
    batches: List[dict],
    device: str = "cpu",
    num_batches: int = 10,
) -> float:
    model.eval()
    model.to(device)
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(batches):
        if i >= num_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            labels.contiguous().view(-1),
        )
        total_loss += loss.item()
        count += 1
    avg_loss = total_loss / max(count, 1)
    return math.exp(min(avg_loss, 20))


def compute_random_baseline_ppl(
    config: TransformerConfig,
    batches: List[dict],
    device: str = "cpu",
    num_batches: int = 10,
) -> float:
    model = MiniGPT2(config).to(device)
    return compute_ppl(model, batches, device, num_batches)


def weight_distance(
    imputed_model: MiniGPT2,
    ground_truth_model: MiniGPT2,
) -> dict:
    mse_w1 = 0.0
    mse_w2 = 0.0
    cos_w1 = 0.0
    cos_w2 = 0.0
    n_layers = imputed_model.config.n_layers

    for i in range(n_layers):
        imp_w1 = imputed_model.blocks[i].mlp.W1.weight
        gt_w1 = ground_truth_model.blocks[i].mlp.W1.weight
        mse_w1 += F.mse_loss(imp_w1, gt_w1).item()
        cos_w1 += F.cosine_similarity(
            imp_w1.view(-1), gt_w1.view(-1), dim=0
        ).item()

        imp_w2 = imputed_model.blocks[i].mlp.W2.weight
        gt_w2 = ground_truth_model.blocks[i].mlp.W2.weight
        mse_w2 += F.mse_loss(imp_w2, gt_w2).item()
        cos_w2 += F.cosine_similarity(
            imp_w2.view(-1), gt_w2.view(-1), dim=0
        ).item()

    return {
        "W1_mse": mse_w1 / n_layers,
        "W2_mse": mse_w2 / n_layers,
        "W1_cosine": cos_w1 / n_layers,
        "W2_cosine": cos_w2 / n_layers,
    }


def mlp_ablation_gap(
    model: MiniGPT2,
    batches: List[dict],
    device: str = "cpu",
    num_batches: int = 10,
) -> dict:
    model.eval()
    model.to(device)
    total_with = 0.0
    total_without = 0.0
    count = 0

    with torch.no_grad():
        for i, batch in enumerate(batches):
            if i >= num_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits_with = model(input_ids, use_mlp=True)
            loss_with = F.cross_entropy(
                logits_with.contiguous().view(-1, logits_with.shape[-1]),
                labels.contiguous().view(-1),
            )

            logits_without = model(input_ids, use_mlp=False)
            loss_without = F.cross_entropy(
                logits_without.contiguous().view(-1, logits_without.shape[-1]),
                labels.contiguous().view(-1),
            )

            total_with += loss_with.item()
            total_without += loss_without.item()
            count += 1

    n = max(count, 1)
    ppl_with = math.exp(min(total_with / n, 20))
    ppl_without = math.exp(min(total_without / n, 20))
    return {
        "ppl_with_mlp": ppl_with,
        "ppl_without_mlp": ppl_without,
        "mlp_gap": ppl_without - ppl_with,
    }


def convergence_head_start(
    imputed_model: MiniGPT2,
    eval_batches: List[dict],
    device: str = "cpu",
    num_steps: int = 200,
    lr: float = 5e-4,
) -> dict:
    imputed_model.to(device)
    imputed_model.train()
    opt_imp = torch.optim.AdamW(imputed_model.parameters(), lr=lr, eps=1e-4)

    random_model = MiniGPT2(imputed_model.config).to(device)
    random_model.train()
    opt_rand = torch.optim.AdamW(random_model.parameters(), lr=lr, eps=1e-4)

    imp_losses = []
    rand_losses = []

    train_batches = load_wikitext_batches(
        batch_size=8, seq_len=1024, split="train", subset_tokens=1024 * 64
    )
    data_iter = wikitext_batch_iterator(train_batches, seed=42)

    for step in range(num_steps):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits_imp = imputed_model(input_ids)
        loss_imp = F.cross_entropy(
            logits_imp.contiguous().view(-1, logits_imp.shape[-1]),
            labels.contiguous().view(-1),
        )

        logits_rand = random_model(input_ids)
        loss_rand = F.cross_entropy(
            logits_rand.contiguous().view(-1, logits_rand.shape[-1]),
            labels.contiguous().view(-1),
        )

        opt_imp.zero_grad()
        loss_imp.backward()
        opt_imp.step()

        opt_rand.zero_grad()
        loss_rand.backward()
        opt_rand.step()

        imp_losses.append(loss_imp.item())
        rand_losses.append(loss_rand.item())

    speedup = num_steps
    for i, loss in enumerate(rand_losses):
        if loss <= imp_losses[0]:
            speedup = i + 1
            break

    return {
        "imputed_initial_loss": imp_losses[0],
        "imputed_final_loss": imp_losses[-1],
        "random_initial_loss": rand_losses[0],
        "random_final_loss": rand_losses[-1],
        "speedup_steps": speedup,
        "speedup_ratio": speedup / num_steps if num_steps > 0 else 0,
    }


def compute_cost_savings(
    d_ff_small: int,
    d_ff_large: int,
    d_model: int,
    n_layers: int,
    seq_len: int = 1024,
) -> dict:
    ratio = (d_ff_large + 2 * d_model) / (d_ff_small + 2 * d_model)
    savings_pct = (1 - 1 / ratio) * 100
    return {
        "d_ff_small": d_ff_small,
        "d_ff_large": d_ff_large,
        "step_time_ratio": round(ratio, 3),
        "savings_pct": round(savings_pct, 1),
    }


def run_full_validation(
    source_path: str,
    ground_truth_path: Optional[str],
    eval_batches: List[dict],
    target_d_ff: int,
    device: str = "cpu",
    num_act_batches: int = 10,
    num_pp_batches: int = 10,
    convergence_steps: int = 200,
) -> dict:
    print(f"Loading source model from {source_path}")
    source_model, src_meta = MiniGPT2.load(source_path, map_location=device)
    source_model.to(device)

    print(f"Imputing MLP to d_ff={target_d_ff}...")
    imputed_model = impute_mlp_to_target(
        source_model, target_d_ff, eval_batches, num_act_batches
    )
    imputed_model.to(device)

    print(f"Source d_ff={source_model.config.d_ff}, target d_ff={target_d_ff}")
    print(f"Parameters: source={source_model.num_parameters():,}, imputed={imputed_model.num_parameters():,}")

    results = {}

    print("\nEvaluating perplexity...")
    results["source_ppl"] = compute_ppl(source_model, eval_batches, device, num_pp_batches)
    results["imputed_ppl"] = compute_ppl(imputed_model, eval_batches, device, num_pp_batches)
    results["random_ppl"] = compute_random_baseline_ppl(
        imputed_model.config, eval_batches, device, num_pp_batches
    )
    print(
        f"  Source PPL: {results['source_ppl']:.2f}\n"
        f"  Imputed PPL: {results['imputed_ppl']:.2f}\n"
        f"  Random PPL: {results['random_ppl']:.2f}"
    )

    if ground_truth_path and os.path.exists(ground_truth_path):
        print("\nComputing weight distance to ground truth...")
        gt_model, gt_meta = MiniGPT2.load(ground_truth_path, map_location=device)
        gt_model.to(device)
        results["ground_truth_ppl"] = compute_ppl(gt_model, eval_batches, device, num_pp_batches)
        results["weight_distance"] = weight_distance(imputed_model, gt_model)
        print(
            f"  GT PPL: {results['ground_truth_ppl']:.2f}\n"
            f"  W1 MSE: {results['weight_distance']['W1_mse']:.6f}\n"
            f"  W2 MSE: {results['weight_distance']['W2_mse']:.6f}\n"
            f"  W1 cos: {results['weight_distance']['W1_cosine']:.4f}\n"
            f"  W2 cos: {results['weight_distance']['W2_cosine']:.4f}"
        )
    else:
        results["ground_truth_ppl"] = None
        results["weight_distance"] = None

    print("\nComputing MLP ablation gap...")
    results["ablation"] = mlp_ablation_gap(source_model, eval_batches, device, num_pp_batches)
    print(
        f"  With MLP: {results['ablation']['ppl_with_mlp']:.2f}\n"
        f"  Without MLP: {results['ablation']['ppl_without_mlp']:.2f}\n"
        f"  Gap: {results['ablation']['mlp_gap']:.2f}"
    )

    print("\nComputing convergence head start...")
    results["convergence"] = convergence_head_start(
        imputed_model, eval_batches, device, convergence_steps
    )
    print(
        f"  Imputed initial loss: {results['convergence']['imputed_initial_loss']:.4f}\n"
        f"  Random initial loss: {results['convergence']['random_initial_loss']:.4f}\n"
        f"  Speedup ratio: {results['convergence']['speedup_ratio']:.3f}"
    )

    cfg = source_model.config
    results["cost_savings"] = compute_cost_savings(
        cfg.d_ff, target_d_ff, cfg.d_model, cfg.n_layers
    )
    print(
        f"\nCost savings: {results['cost_savings']['savings_pct']}% "
        f"(step time ratio: {results['cost_savings']['step_time_ratio']})"
    )

    results["source_meta"] = src_meta
    return results


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="experiments/dff_variation/config.yaml"
    )
    parser.add_argument("--source-path", type=str, required=True)
    parser.add_argument("--ground-truth-path", type=str, default=None)
    parser.add_argument("--output", type=str, default="dff_validation_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-act-batches", type=int, default=10)
    parser.add_argument("--num-pp-batches", type=int, default=10)
    parser.add_argument("--convergence-steps", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    target_d_ff = config["d_ff_variation"]["target_d_ff"]
    batch_size = config["training"]["batch_size"]
    seq_len = config["training"]["seq_len"]

    print("Loading evaluation data...")
    eval_batches = load_wikitext_batches(
        batch_size=batch_size,
        seq_len=seq_len,
        split="validation",
        device=args.device,
    )
    print(f"  {len(eval_batches)} eval batches ready")

    results = run_full_validation(
        source_path=args.source_path,
        ground_truth_path=args.ground_truth_path,
        eval_batches=eval_batches,
        target_d_ff=target_d_ff,
        device=args.device,
        num_act_batches=args.num_act_batches,
        num_pp_batches=args.num_pp_batches,
        convergence_steps=args.convergence_steps,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    _main()