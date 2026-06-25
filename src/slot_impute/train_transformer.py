import math
import os
import time
from typing import Optional

import torch
import torch.nn.functional as F

from .data import load_wikitext_batches, wikitext_batch_iterator
from .transformer import MiniGPT2, TransformerConfig


def train_model(
    model: MiniGPT2,
    batches: list,
    device: str = "cpu",
    steps: int = 20000,
    lr: float = 6e-4,
    seed: int = 42,
    log_interval: int = 100,
    save_dir: Optional[str] = None,
    checkpoint_tag: str = "model",
    warmup_steps: int = 1000,
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> list:
    torch.manual_seed(seed)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.startswith("cuda") else None

    data_iter = wikitext_batch_iterator(batches, seed=seed)
    loss_history = []
    t0 = time.time()

    for step in range(steps):
        if step < warmup_steps:
            lr_now = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast("cuda"):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.contiguous().view(-1, logits.shape[-1]),
                    labels.contiguous().view(-1),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.contiguous().view(-1, logits.shape[-1]),
                labels.contiguous().view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        loss_history.append(loss.item())

        if log_interval and (step + 1) % log_interval == 0:
            avg_loss = sum(loss_history[-log_interval:]) / log_interval
            ppl = math.exp(min(avg_loss, 20))
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  Step {step + 1}/{steps}: loss={avg_loss:.4f} ppl={ppl:.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"steps/s={steps_per_sec:.1f}"
            )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        final_window = min(100, len(loss_history))
        avg_final = sum(loss_history[-final_window:]) / final_window
        metadata = {
            "seed": seed,
            "steps": steps,
            "d_ff": model.config.d_ff,
            "n_params": model.num_parameters(),
            "final_loss": avg_final,
            "final_ppl": math.exp(min(avg_final, 20)),
            "loss_history": loss_history,
            "train_time_s": time.time() - t0,
        }
        path = os.path.join(
            save_dir, f"dff{model.config.d_ff}_seed{seed}_{checkpoint_tag}.pt"
        )
        MiniGPT2.save(model, path, metadata)
        print(
            f"  Saved {path} (loss={avg_final:.4f}, ppl={metadata['final_ppl']:.2f})"
        )

    return loss_history


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="experiments/dff_variation/config.yaml"
    )
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints_dff")
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--subset-tokens", type=int, default=None)
    parser.add_argument("--use-amp", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    d_ff = args.d_ff or config["d_ff_variation"]["source_d_ff"]
    steps = args.steps or config["training"]["steps"]
    lr = args.lr or config["training"]["lr"]
    batch_size = args.batch_size or config["training"]["batch_size"]
    seq_len = args.seq_len or config["training"]["seq_len"]

    if args.quick_test:
        d_ff = config["d_ff_variation"]["source_d_ff"]
        steps = config["d_ff_variation"]["quick_test_steps"]
        subset = args.subset_tokens or 1024 * 32
    else:
        subset = args.subset_tokens

    mc = config["model"]
    model_cfg = TransformerConfig(
        vocab_size=mc["vocab_size"],
        d_model=mc["d_model"],
        d_ff=d_ff,
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        max_seq_len=mc["max_seq_len"],
    )
    model = MiniGPT2(model_cfg)
    print(
        f"Model: d_model={model_cfg.d_model} d_ff={d_ff} n_layers={model_cfg.n_layers} "
        f"params={model.num_parameters():,}"
    )

    print(f"Loading Wikitext-103 ({'subset=' + str(subset) if subset else 'full'})...")
    batches = load_wikitext_batches(
        batch_size=batch_size,
        seq_len=seq_len,
        split="train",
        subset_tokens=subset,
        device=args.device,
    )
    print(f"  {len(batches)} batches ready")

    print(f"Training d_ff={d_ff} seed={args.seed} steps={steps} device={args.device}")
    train_model(
        model,
        batches,
        device=args.device,
        steps=steps,
        lr=lr,
        seed=args.seed,
        save_dir=args.save_dir,
        checkpoint_tag=args.tag,
        warmup_steps=config["training"]["warmup_steps"],
        grad_clip=config["training"]["grad_clip"],
        use_amp=args.use_amp,
    )


if __name__ == "__main__":
    _main()