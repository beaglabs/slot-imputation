import json
import math
import os
from typing import Callable, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reference import slot_murmurate_reference


LITHOLOGY_NAMES = [
    "shale", "siltstone", "sandstone", "conglomerate",
    "limestone", "dolostone", "marl", "evaporite",
    "coal", "chert", "basalt", "granite",
]


def make_geology_transition_matrix(seed: int = 42) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    N = len(LITHOLOGY_NAMES)
    raw = torch.zeros(N, N)

    idx = {name: i for i, name in enumerate(LITHOLOGY_NAMES)}

    shale = idx["shale"]
    silt = idx["siltstone"]
    sand = idx["sandstone"]
    cong = idx["conglomerate"]
    lime = idx["limestone"]
    dolo = idx["dolostone"]
    marl = idx["marl"]
    evap = idx["evaporite"]
    coal = idx["coal"]
    chert = idx["chert"]
    basalt = idx["basalt"]
    gran = idx["granite"]

    self_stick = 0.35

    def row(i, targets):
        total = 0.0
        for j, w in targets.items():
            raw[i, j] = w
            total += w
        raw[i, i] = self_stick
        rs = raw[i].sum()
        raw[i] = raw[i] / rs

    row(shale, {silt: 0.30, marl: 0.18, coal: 0.12, sand: 0.05})
    row(silt, {shale: 0.20, sand: 0.30, marl: 0.10, coal: 0.05})
    row(sand, {silt: 0.22, cong: 0.28, shale: 0.08, lime: 0.07})
    row(cong, {sand: 0.35, shale: 0.10, gran: 0.10, basalt: 0.05, lime: 0.05})
    row(lime, {dolo: 0.25, marl: 0.18, chert: 0.12, shale: 0.05, sand: 0.05})
    row(dolo, {lime: 0.25, evap: 0.20, marl: 0.10, shale: 0.05, chert: 0.05})
    row(marl, {lime: 0.25, shale: 0.25, dolo: 0.10, silt: 0.05})
    row(evap, {dolo: 0.30, lime: 0.15, marl: 0.10, shale: 0.10})
    row(coal, {shale: 0.35, silt: 0.15, sand: 0.10, lime: 0.05})
    row(chert, {lime: 0.30, dolo: 0.20, shale: 0.10, sand: 0.05})
    row(basalt, {gran: 0.20, cong: 0.15, sand: 0.15, shale: 0.10, lime: 0.05})
    row(gran, {basalt: 0.20, cong: 0.20, sand: 0.15, shale: 0.05, lime: 0.05})

    return raw


def build_geology_data(
    seq_len: int,
    num_batches: int,
    corruption_rate: float = 0.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.long,
    seed: int = 42,
    chain_seed: int = 42,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    N = len(LITHOLOGY_NAMES)
    trans = make_geology_transition_matrix(seed=chain_seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    for _ in range(num_batches):
        seq = torch.zeros(1, seq_len, device=device, dtype=dtype)
        seq[0, 0] = torch.randint(0, N, (1,), device=device, generator=gen).item()
        for t in range(1, seq_len):
            probs = trans[seq[0, t - 1].item()]
            seq[0, t] = torch.multinomial(probs, 1, generator=gen).item()
        clean_ids = seq

        corrupted = clean_ids.clone()
        if corruption_rate > 0:
            mask = torch.rand(1, seq_len, device=device, generator=gen) < corruption_rate
            n_masked = mask.sum().item()
            if n_masked > 0:
                noise = torch.randint(0, N, (n_masked,), device=device, dtype=dtype, generator=gen)
                corrupted[mask] = noise
        yield corrupted, clean_ids


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

    def forward(self, input_ids: torch.Tensor, use_slots: bool = True) -> torch.Tensor:
        x = self.embed(input_ids)
        B, N, D = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if use_slots:
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
        else:
            out = self.output_proj(x)

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
        seq[0, 0] = torch.randint(0, num_states, (1,), device=device, generator=gen).item()
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
    parser.add_argument("--task", type=str, default="random", choices=["random", "structured", "geology"])
    parser.add_argument("--num-states", type=int, default=16)
    parser.add_argument("--chain-seed", type=int, default=42)
    parser.add_argument("--num-lithologies", type=int, default=12)
    parser.add_argument("--geology-seed", type=int, default=42)
    args = parser.parse_args()

    if args.task == "geology":
        vocab_for_task = args.num_lithologies
        task_label = f"task=geology lithologies={args.num_lithologies}"
    elif args.task == "structured":
        vocab_for_task = args.num_states
        task_label = f"task=structured states={args.num_states}"
    else:
        vocab_for_task = 256
        task_label = "random"

    model = MurmurativeProbe(num_slots=args.num_slots, vocab_size=vocab_for_task)
    print(f"Training M={args.num_slots} seed={args.seed} seq_len={args.seq_len} corr={args.corruption_rate} {task_label} on {args.device}")

    if args.task == "geology":
        data_fn = lambda: build_geology_data(
            seq_len=args.seq_len,
            num_batches=args.steps,
            corruption_rate=args.corruption_rate,
            device=args.device,
            seed=args.seed,
            chain_seed=args.geology_seed,
        )
    elif args.task == "structured":
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