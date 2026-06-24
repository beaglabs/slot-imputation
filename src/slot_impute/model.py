import json
import math
import os
from typing import Callable, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference import slot_murmurate_reference


class MurmurativeProbe(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 512,
        num_heads: int = 8,
        num_slots: int = 256,
        num_rounds: int = 3,
        alpha: float = 0.9,
        gamma: float = 0.15,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_slots = num_slots
        self.num_rounds = num_rounds
        self.alpha = alpha
        self.gamma = gamma

        Dh = d_model // num_heads

        self.embed = nn.Embedding(vocab_size, d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.slot_k_emb = nn.Parameter(torch.randn(num_heads, num_slots, Dh) * 0.02)
        self.slot_v_emb = nn.Parameter(torch.randn(num_heads, num_slots, Dh) * 0.02)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        B, N, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        slot_out = slot_murmurate_reference(
            x=x,
            q_proj=q,
            k_proj=k,
            v_proj=v,
            slot_k_emb=self.slot_k_emb,
            slot_v_emb=self.slot_v_emb,
            num_heads=self.num_heads,
            rounds=self.num_rounds,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        out = self.output_proj(slot_out)
        logits = self.lm_head(out)
        return logits

    @staticmethod
    def save(model: "MurmurativeProbe", path: str, metadata: dict) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        config = {
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "num_heads": model.num_heads,
            "num_slots": model.num_slots,
            "num_rounds": model.num_rounds,
            "alpha": model.alpha,
            "gamma": model.gamma,
        }
        data = {
            "state_dict": model.state_dict(),
            "config": config,
            "metadata": metadata,
        }
        torch.save(data, path)

    @staticmethod
    def load(path: str, map_location: str = "cpu") -> Tuple["MurmurativeProbe", dict]:
        data = torch.load(path, map_location=map_location, weights_only=False)
        config = data["config"]
        model = MurmurativeProbe(**config)
        model.load_state_dict(data["state_dict"])
        return model, data["metadata"]


def build_synthetic_data(
    vocab_size: int,
    seq_len: int,
    num_batches: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.long,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=dtype)
        yield input_ids, input_ids.clone()


def build_corrupted_data(
    vocab_size: int,
    seq_len: int,
    num_batches: int,
    corruption_rate: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.long,
    seed: int = 42,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for _ in range(num_batches):
        clean_ids = torch.randint(0, vocab_size, (1, seq_len), device=device, dtype=dtype, generator=generator)
        corrupted = clean_ids.clone()
        if corruption_rate > 0:
            mask = torch.rand(1, seq_len, device=device, generator=generator) < corruption_rate
            n_masked = mask.sum().item()
            if n_masked > 0:
                noise = torch.randint(0, vocab_size, (n_masked,), device=device, dtype=dtype, generator=generator)
                corrupted[mask] = noise
        yield corrupted, clean_ids


def make_markov_chain(
    num_states: int,
    seed: int = 42,
) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    raw = torch.rand(num_states, num_states, generator=g)
    raw = raw + torch.eye(num_states) * 2.0
    return raw / raw.sum(dim=1, keepdim=True)


def build_markov_data(
    num_states: int,
    seq_len: int,
    num_batches: int,
    corruption_rate: float = 0.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.long,
    seed: int = 42,
    chain_seed: int = 42,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    trans = make_markov_chain(num_states, seed=chain_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    for _ in range(num_batches):
        seq = torch.zeros(1, seq_len, device=device, dtype=dtype)
        seq[0, 0] = torch.randint(0, num_states, (1,), generator=gen).item()
        for t in range(1, seq_len):
            probs = trans[seq[0, t - 1].item()]
            seq[0, t] = torch.multinomial(probs, 1, generator=gen).item()
        clean_ids = seq

        corrupted = clean_ids.clone()
        if corruption_rate > 0:
            mask = torch.rand(1, seq_len, device=device, generator=gen) < corruption_rate
            n_masked = mask.sum().item()
            if n_masked > 0:
                noise = torch.randint(0, num_states, (n_masked,), device=device, dtype=dtype, generator=gen)
                corrupted[mask] = noise
        yield corrupted, clean_ids


def train_model(
    model: "MurmurativeProbe",
    device: str = "cpu",
    steps: int = 500,
    lr: float = 5e-4,
    seed: int = 42,
    seq_len: int = 2048,
    corruption_rate: float = 0.25,
    log_interval: int = 100,
    save_dir: str = None,
    data_fn: Optional[Callable[[], Iterator[Tuple[torch.Tensor, torch.Tensor]]]] = None,
) -> list[float]:
    torch.manual_seed(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-4)
    loss_history = []

    if data_fn is not None:
        data_iter = data_fn()
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    for step in range(steps):
        if data_fn is not None:
            try:
                corrupted_ids, clean_ids = next(data_iter)
            except StopIteration:
                data_iter = data_fn()
                corrupted_ids, clean_ids = next(data_iter)
        else:
            clean_ids = torch.randint(
                0, model.vocab_size, (1, seq_len), device=device, dtype=torch.long, generator=generator
            )
            corrupted_ids = clean_ids.clone()
            if corruption_rate > 0:
                corruption_mask = (
                    torch.rand(1, seq_len, device=device, generator=generator) < corruption_rate
                )
                num_corrupt = corruption_mask.sum().item()
                if num_corrupt > 0:
                    noise = torch.randint(
                        0, model.vocab_size, (num_corrupt,), device=device, dtype=torch.long, generator=generator
                    )
                    corrupted_ids[corruption_mask] = noise

        optimizer.zero_grad()
        logits = model(corrupted_ids)
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size), clean_ids.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(loss.item())

        if log_interval and (step + 1) % log_interval == 0:
            avg = sum(loss_history[-log_interval:]) / log_interval
            print(f"  Step {step + 1}/{steps}: loss={avg:.4f}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        avg_final = sum(loss_history[-100:]) / min(100, len(loss_history))
        metadata = {
            "seed": seed,
            "final_loss": avg_final,
            "final_ppl": math.exp(min(avg_final, 20)),
            "loss_history": loss_history,
        }
        path = os.path.join(save_dir, f"M{model.num_slots}_seed{seed}.pt")
        MurmurativeProbe.save(model, path, metadata)
        print(f"  Saved {path} (final_loss={avg_final:.4f}, ppl={metadata['final_ppl']:.2f})")

    return loss_history


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-slots", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--corruption-rate", type=float, default=0.25)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--task", type=str, default="random", choices=["random", "structured"])
    parser.add_argument("--num-states", type=int, default=16)
    parser.add_argument("--chain-seed", type=int, default=42)
    args = parser.parse_args()

    vocab_for_task = args.num_states if args.task == "structured" else 256
    model = MurmurativeProbe(num_slots=args.num_slots, vocab_size=vocab_for_task)
    task_label = f"task={args.task} states={args.num_states}" if args.task == "structured" else "random"
    print(f"Training M={args.num_slots} seed={args.seed} seq_len={args.seq_len} corr={args.corruption_rate} {task_label} on {args.device}")

    if args.task == "structured":
        data_fn = lambda: build_markov_data(
            num_states=args.num_states, seq_len=args.seq_len,
            num_batches=args.steps, corruption_rate=args.corruption_rate,
            device=args.device, seed=args.seed, chain_seed=args.chain_seed,
        )
    else:
        data_fn = None

    train_model(
        model,
        device=args.device,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        seq_len=args.seq_len,
        corruption_rate=args.corruption_rate,
        save_dir=args.save_dir,
        data_fn=data_fn,
    )


if __name__ == "__main__":
    _main()