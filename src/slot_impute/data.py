from typing import Iterator, List, Optional

import torch


def load_wikitext_batches(
    batch_size: int = 8,
    seq_len: int = 1024,
    split: str = "train",
    subset_tokens: Optional[int] = None,
    device: str = "cpu",
) -> List[dict]:
    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-103-v1", split=split)
    dataset = dataset.filter(lambda x: len(x["text"]) > 0)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            return_tensors=None,
            add_special_tokens=False,
            truncation=False,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    all_ids = []
    for example in dataset:
        all_ids.extend(example["input_ids"])

    if subset_tokens:
        all_ids = all_ids[:subset_tokens]

    total = len(all_ids)
    overlap = seq_len
    stride = seq_len

    chunks = []
    for start in range(0, total - overlap, stride):
        end = min(start + seq_len + 1, total)
        chunk = all_ids[start:end]
        if len(chunk) < seq_len + 1:
            break
        chunks.append(chunk)

    batched = []
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        batch_chunks = chunks[i : i + batch_size]
        tensor = torch.tensor(batch_chunks, dtype=torch.long, device=device)
        batched.append({"input_ids": tensor[:, :-1], "labels": tensor[:, 1:]})

    return batched


def wikitext_batch_iterator(
    batches: List[dict],
    seed: int = 42,
) -> Iterator[dict]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    indices = torch.randperm(len(batches), generator=rng).tolist()

    while True:
        for idx in indices:
            yield batches[idx]