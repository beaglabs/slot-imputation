# Slot Impute Experiment

Test whether slot pool weights can be interpolated across the M (slot count)
dimension, enabling "auto-fill" of a target model's slot pool from cheaper-to-train
anchor models. Validates the kriging + interferometry approach at minimal cost,
runnable on an M2 Mac with no GPU.

---

## 1. Repo Structure

```
slot-impute/
├── pyproject.toml
├── README.md
├── src/
│   └── slot_impute/
│       ├── __init__.py
│       ├── reference.py            # Copied from murmurative-attention (192 lines, pure torch)
│       ├── model.py                # MurmurativeProbe with checkpoint save/load
│       ├── extract.py              # extract/inject slot pool weights
│       ├── interferometry.py       # Hungarian alignment + signal/noise mask
│       ├── variogram.py            # linear, spline, kriging interpolators
│       ├── impute.py               # impute target model from anchors
│       ├── validate.py             # ppl, MSE, cosine sim, convergence speedup
│       └── report.py               # markdown report + tables
├── scripts/
│   ├── train_anchors.sh            # train all 15 models in parallel
│   └── run_pipeline.sh             # end-to-end: train -> impute -> validate -> report
└── experiments/
    └── m_variation/
        └── config.yaml             # M values, seeds, steps, training hyperparams
```

---

## 2. Dependencies

`pyproject.toml`:
```toml
[project]
name = "slot-impute"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.5",
    "numpy",
    "scipy",         # Hungarian algorithm, spline interpolation
    "pyyaml",        # config parsing
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

No CUDA. No murmurative-kernels. Fully self-contained.

---

## 3. Experimental Design

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| `num_heads` (H) | 8 |
| `Dh` | 64 |
| `seq_len` | 128 |
| `vocab_size` | 256 |
| `rounds` (R) | 3 |
| `training_steps` | 500 |
| `lr` | 5e-4 |

**Varied**:

| M (slot count) | Seeds | Role |
|---|---|---|
| 128 | 42, 123, 999 | Anchor |
| 192 | 42, 123, 999 | Anchor |
| 256 | 42, 123, 999 | **Ground truth** (held out) |
| 384 | 42, 123, 999 | Anchor |
| 512 | 42, 123, 999 | Anchor |
| **Total: 15 models** | | |

All models share identical shapes for QKV projections, embedding, output_proj,
and lm_head. Only `slot_k_emb [8, M, 64]` and `slot_v_emb [8, M, 64]` change
with M.

---

## 4. File-by-File Spec

### 4.1 `src/slot_impute/reference.py`

Copy `src/murmurative/reference.py` from murmurative-attention verbatim (192 lines).
It implements `slot_select_reference`, `slot_attend_reference`,
`slot_update_reference`, `slot_diffusion_reference`, and `slot_murmurate_reference`
using pure PyTorch operations (einsum, topk, gather, scatter_add, softmax).

No modifications needed.

### 4.2 `src/slot_impute/model.py`

`MurmurativeProbe` class — identical architecture to `train_probe.py` but with
checkpoint I/O.

```python
class MurmurativeProbe(nn.Module):
    def __init__(self, vocab_size=256, d_model=512, num_heads=8,
                 num_slots=256, num_rounds=3, alpha=0.9, gamma=0.15)
        # embed, q_proj, k_proj, v_proj, slot_k_emb, slot_v_emb,
        # output_proj, lm_head

    def forward(self, input_ids)
        # x = embed(input_ids)
        # qkv projections
        # slot_murmurate_reference(...)
        # output_proj -> lm_head -> logits

    @staticmethod
    def save(model, path, metadata)
        # torch.save({weights, config, metadata}, path)

    @staticmethod
    def load(path, map_location="cpu")
        # returns (model, metadata)
```

**Key details**:
- `save()` stores a dict: `{"state_dict": model.state_dict(), "config": {...}, "metadata": {"seed": int, "final_loss": float, "final_ppl": float, "loss_history": [float]}}`
- `load()` reconstructs the model from config, loads state dict, returns `(model, metadata)`
- Both use `torch.save` / `torch.load` — no safetensors needed at this scale

**Training loop** (a function, not a class method — called from scripts):
```python
def train_model(model, device, steps=500, lr=5e-4, seed=42,
                log_interval=100, save_dir=None):
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-4)
    loss_history = []

    for step, (input_ids, target_ids) in enumerate(data_iter):
        optimizer.zero_grad()
        logits = model(input_ids.to(device))
        loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(loss.item())

    if save_dir:
        avg_final = sum(loss_history[-100:]) / min(100, len(loss_history))
        MurmurativeProbe.save(model, f"{save_dir}/M{num_slots}_seed{seed}.pt",
                              {"seed": seed, "final_loss": avg_final,
                               "final_ppl": math.exp(min(avg_final, 20)),
                               "loss_history": loss_history})
    return loss_history
```

Synthetic data generator: identical to `train_probe.py` — `torch.randint(0, 256, (1, 128))`
for each step. Deterministic per seed via `torch.manual_seed(seed)`.

### 4.3 `src/slot_impute/extract.py`

```python
def load_checkpoint(path) -> Tuple[nn.Module, dict]
    # Returns (model, metadata)

def extract_slot_pool(model) -> Tuple[torch.Tensor, torch.Tensor]
    # Returns (slot_k_emb, slot_v_emb) as detached tensors
    # Shapes: [H, M, Dh]

def extract_all_weights(model) -> dict
    # Returns dict of all learnable parameters as detached tensors:
    #   {"embed": [V, D], "q_proj": [D, D], "k_proj": [D, D],
    #    "v_proj": [D, D], "slot_k": [H, M, Dh], "slot_v": [H, M, Dh],
    #    "output_proj": [D, D], "lm_head": [D, V]}

def inject_slot_pool(model, slot_k, slot_v)
    # Replaces model.slot_k_emb.data and model.slot_v_emb.data with given tensors
    # Returns model for chaining

def inject_all_weights(model, weights_dict)
    # Replaces all learnable parameters from weights_dict
```

### 4.4 `src/slot_impute/interferometry.py`

For a single M value, aligns slot pools across 3 seeds and computes a
signal/noise mask.

```python
def hungarian_align_slots(slot_k_a, slot_k_b):
    """
    Align slot ordering of model B to match model A.
    - slot_k_a, slot_k_b: [H, M, Dh]
    - For each head, compute MxM cosine similarity matrix
    - Run Hungarian algorithm (scipy.optimize.linear_sum_assignment) to find
      optimal matching
    - Returns: permutation vector [M] for each head, and aligned slot_k_b, slot_v_b
    """
    H, M, Dh = slot_k_a.shape
    perm = torch.zeros(H, M, dtype=torch.long)
    for h in range(H):
        # Normalize for cosine similarity
        a = F.normalize(slot_k_a[h], dim=-1)  # [M, Dh]
        b = F.normalize(slot_k_b[h], dim=-1)
        sim = a @ b.T  # [M, M]
        cost = -sim.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        perm[h] = torch.tensor(col_ind)
    return perm
```

```python
def align_all_seeds(checkpoints):
    """
    Load all checkpoints for one M value, align to seed 42.
    Returns: list of (slot_k_aligned, slot_v_aligned) tensors, one per seed.
    """
    # Load seed 42 as reference
    ref_k, ref_v = extract_slot_pool(load_checkpoint(checkpoints[42])[0])

    aligned_k = [ref_k]
    aligned_v = [ref_v]

    for seed in [123, 999]:
        model = load_checkpoint(checkpoints[seed])[0]
        sk, sv = extract_slot_pool(model)
        perm = hungarian_align_slots(ref_k, sk)
        # Apply permutation to sk, sv for each head
        sk_aligned = sk.clone()
        sv_aligned = sv.clone()
        for h in range(H):
            sk_aligned[h] = sk[h, perm[h]]
            sv_aligned[h] = sv[h, perm[h]]
        aligned_k.append(sk_aligned)
        aligned_v.append(sv_aligned)

    return aligned_k, aligned_v
```

```python
def compute_signal_mask(aligned_slot_k, aligned_slot_v, threshold="median"):
    """
    Compute per-weight variance across seeds.
    aligned_slot_k: list of [H, M, Dh] tensors (one per seed)
    Returns: signal_mask [H, M, Dh] (bool), variance [H, M, Dh] (float)
    """
    stacked_k = torch.stack(aligned_slot_k)  # [3, H, M, Dh]
    stacked_v = torch.stack(aligned_slot_v)

    var_k = stacked_k.var(dim=0)  # [H, M, Dh]
    var_v = stacked_v.var(dim=0)
    variance = var_k + var_v  # combined variance

    if threshold == "median":
        thresh = variance.median()
    elif isinstance(threshold, float):
        thresh = threshold
    else:
        raise ValueError(f"Unknown threshold: {threshold}")

    signal_mask = variance < thresh
    return signal_mask, variance
```

```python
def interferometry_report(alignments_by_M):
    """
    For each M, report:
    - Signal ratio (fraction of weights below threshold)
    - Mean/median variance of signal vs noise weights
    - Does signal ratio change with M?
    Returns: dict with per-M stats
    """
```

### 4.5 `src/slot_impute/variogram.py`

Models how slot weights change with M and predicts weights at target M.

**Core concept**: Slot index `i` at M maps to normalized position `p = i / (M-1)` in [0, 1].
For a target M, slot `j` maps to `p_j = j / (M_t - 1)`. Anchors at different M values
are sampled at the same normalized positions.

```python
def normalize_positions(M):
    """Returns tensor of positions [0, 1/(M-1), 2/(M-1), ..., 1] of shape [M]"""
    return torch.linspace(0, 1, M)

def sample_at_positions(slot_weights, M, target_positions):
    """
    slot_weights: [H, M, Dh]
    target_positions: [N_target] in [0, 1]
    Returns: [H, N_target, Dh] — weights interpolated to target positions
    Uses linear interpolation along M axis.
    """
    src_pos = normalize_positions(M)  # [M]
    H, _, Dh = slot_weights.shape
    N_t = len(target_positions)

    # For each target position, linear interpolate between two nearest source positions
    # Use torch.searchsorted or manual interpolation
    result = torch.zeros(H, N_t, Dh)
    for i, p in enumerate(target_positions):
        # Find bracketing source indices
        idx = torch.searchsorted(src_pos, p)
        idx = idx.clamp(1, M - 1)
        lo, hi = idx - 1, idx
        alpha = (p - src_pos[lo]) / (src_pos[hi] - src_pos[lo] + 1e-8)
        result[:, i] = (1 - alpha) * slot_weights[:, lo] + alpha * slot_weights[:, hi]
    return result
```

**Three interpolators**:

```python
def interpolate_linear(anchor_weights, anchor_Ms, target_M, target_positions):
    """
    anchor_weights: list of [H, M_a, Dh] for each anchor
    anchor_Ms: list of M values
    target_M: target slot count
    target_positions: positions to predict at

    For each target position, fit line w(M) = a*p + b through anchor points,
    predict at target_M.
    """
    N_t = len(target_positions)
    H, _, Dh = anchor_weights[0].shape

    # Sample all anchors at target_positions
    sampled = []  # list of [H, N_t, Dh]
    for w, M in zip(anchor_weights, anchor_Ms):
        sampled.append(sample_at_positions(w, M, target_positions))

    # Now for each (h, pos, dim) we have values at different M
    # Fit linear regression: w(M) = a * M + b
    stacked = torch.stack(sampled, dim=0)  # [A, H, N_t, Dh]
    Ms_tensor = torch.tensor(anchor_Ms, dtype=torch.float32)  # [A]

    # Mean and std of anchors
    mean_w = stacked.mean(dim=0)  # [H, N_t, Dh]
    Ms_centered = Ms_tensor - Ms_tensor.mean()

    # Covariance and slope
    cov = ((stacked - mean_w) * Ms_centered.view(-1, 1, 1, 1)).sum(dim=0)
    var_M = (Ms_centered ** 2).sum()
    slope = cov / (var_M + 1e-8)

    # Predict at target_M
    intercept = mean_w - slope * Ms_tensor.mean()
    predicted = slope * target_M + intercept
    return predicted  # [H, N_t, Dh]
```

```python
def interpolate_spline(anchor_weights, anchor_Ms, target_M, target_positions):
    """
    Cubic spline through anchor points. Use scipy.interpolate.CubicSpline
    for each (head, position, dim) independently.
    Falls back to linear if scipy unavailable.
    """
    # For each (h, pos, dim), extract values at each anchor M
    # Fit CubicSpline, predict at target_M
```

```python
def interpolate_krige(anchor_weights, anchor_Ms, target_M, target_positions):
    """
    Ordinary kriging with an exponential variogram model:
        gamma(h) = sill * (1 - exp(-h / range_))

    h = |M_i - M_j|  (distance between anchor sizes in slots)

    For each (head, position, dim):
    1. Compute empirical semivariance between all anchor pairs
    2. Fit sill and range parameters
    3. Solve kriging system for weights
    4. Predict at target_M with uncertainty estimate

    Returns: (predicted [H, N_t, Dh], variance [H, N_t, Dh])
    """
```

### 4.6 `src/slot_impute/impute.py`

Assembles imputed models from anchor checkpoints.

```python
def build_imputed_model(target_M, anchor_checkpoints, anchor_Ms,
                        interpolation_method="linear",
                        signal_masks=None,
                        use_signal_only=False,
                        non_slot_strategy="average"):
    """
    Build a complete MurmurativeProbe with imputed slot pool.

    target_M: e.g., 256
    anchor_checkpoints: list of paths to .pt files
    anchor_Ms: list of M values corresponding to checkpoints
    interpolation_method: "linear", "spline", or "krige"
    signal_masks: dict {M: signal_mask} from interferometry (optional)
    use_signal_only: if True, only interpolate signal weights, fill noise with mean
    non_slot_strategy: "average" or "best" (use weights from lowest-ppl anchor)

    Returns: (model, metadata)
    """
    # 1. Load and extract all anchor weights
    all_weights = []
    for path in anchor_checkpoints:
        model, meta = load_checkpoint(path)
        all_weights.append((extract_all_weights(model), meta))

    # 2. Compute non-slot weights (embed, QKV, output_proj, lm_head)
    if non_slot_strategy == "average":
        non_slot = average_non_slot_weights(all_weights)
    else:  # "best"
        best = min(all_weights, key=lambda x: x[1]["final_ppl"])
        non_slot = best[0]  # use the best anchor's weights

    del non_slot["slot_k"], non_slot["slot_v"]  # we'll impute these

    # 3. Krige slot pool
    slot_k_list = [w["slot_k"] for w, _ in all_weights]  # list of [H, M_a, Dh]
    slot_v_list = [w["slot_v"] for w, _ in all_weights]

    target_positions = normalize_positions(target_M)

    if interpolation_method == "linear":
        slot_k_pred = interpolate_linear(slot_k_list, anchor_Ms, target_M, target_positions)
        slot_v_pred = interpolate_linear(slot_v_list, anchor_Ms, target_M, target_positions)
    elif interpolation_method == "spline":
        slot_k_pred = interpolate_spline(slot_k_list, anchor_Ms, target_M, target_positions)
        slot_v_pred = interpolate_spline(slot_v_list, anchor_Ms, target_M, target_positions)
    elif interpolation_method == "krige":
        slot_k_pred, krige_var_k = interpolate_krige(slot_k_list, anchor_Ms, target_M, target_positions)
        slot_v_pred, krige_var_v = interpolate_krige(slot_v_list, anchor_Ms, target_M, target_positions)

    # 4. Apply signal mask if provided
    if signal_masks is not None and use_signal_only:
        # Interpolate signal mask to target M
        target_mask = interpolate_mask(signal_masks, anchor_Ms, target_M)
        # For noise slots, use mean of all anchors
        mean_k = torch.stack([sample_at_positions(sk, M, target_positions) for sk, M in zip(slot_k_list, anchor_Ms)]).mean(0)
        mean_v = torch.stack([sample_at_positions(sv, M, target_positions) for sv, M in zip(slot_v_list, anchor_Ms)]).mean(0)
        slot_k_pred = torch.where(target_mask, slot_k_pred, mean_k)
        slot_v_pred = torch.where(target_mask, slot_v_pred, mean_v)

    # 5. Build model
    model = MurmurativeProbe(num_slots=target_M)
    inject_all_weights(model, non_slot)
    inject_slot_pool(model, slot_k_pred, slot_v_pred)

    return model, {"slot_k_pred": slot_k_pred, "slot_v_pred": slot_v_pred,
                   "krige_var_k": krige_var_k if interpolation_method == "krige" else None,
                   "krige_var_v": krige_var_v if interpolation_method == "krige" else None}
```

```python
def build_imputed_variants(anchor_checkpoints, anchor_Ms, target_M=256):
    """
    Build all imputation variants:
    A: All-4 anchors, linear interpolation, non_slot=average
    B: Boundary only (M=128, 512), linear, non_slot=average
    C: All-4 anchors, linear, signal-only fill
    D: Boundary only, linear, non_slot=best (naive baseline)

    Returns: dict of {variant_name: (model, metadata)}
    """
    variants = {}

    # A: Full method
    variants["A_full"] = build_imputed_model(
        target_M, anchor_checkpoints, anchor_Ms,
        interpolation_method="linear",
        non_slot_strategy="average"
    )

    # B: Boundary only
    boundary_checkpoints = [cp for cp, M in zip(anchor_checkpoints, anchor_Ms) if M in [128, 512]]
    boundary_Ms = [M for M in anchor_Ms if M in [128, 512]]
    variants["B_boundary"] = build_imputed_model(
        target_M, boundary_checkpoints, boundary_Ms,
        interpolation_method="linear",
        non_slot_strategy="average"
    )

    # C: Signal-only
    variants["C_signal"] = build_imputed_model(
        target_M, anchor_checkpoints, anchor_Ms,
        interpolation_method="linear",
        signal_masks=... ,  # from interferometry
        use_signal_only=True,
        non_slot_strategy="average"
    )

    # D: Naive baseline
    variants["D_naive"] = build_imputed_model(
        target_M, boundary_checkpoints, boundary_Ms,
        interpolation_method="linear",
        non_slot_strategy="best"
    )

    return variants
```

### 4.7 `src/slot_impute/validate.py`

```python
def compute_perplexity(model, data_iter, device="cpu", num_batches=10):
    """
    Zero-shot perplexity on fresh data batches.
    Returns: average perplexity across num_batches.
    """
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for input_ids, target_ids in data_iter:
            logits = model(input_ids.to(device))
            loss = F.cross_entropy(logits.view(-1, 256), target_ids.view(-1).to(device))
            total_loss += loss.item()
            count += 1
            if count >= num_batches:
                break
    avg_loss = total_loss / count
    return math.exp(min(avg_loss, 20))
```

```python
def weight_distance(imputed_model, ground_truth_model):
    """
    Computes MSE and cosine similarity between slot pools.
    Before comparing, alignment via Hungarian may be needed if slot orderings differ.
    Returns: {"slot_k_mse": float, "slot_k_cosine": float,
              "slot_v_mse": float, "slot_v_cosine": float}
    """
```

```python
def convergence_speedup(imputed_model, random_init_model, num_steps=100,
                        device="cpu"):
    """
    Fine-tune imputed model for num_steps.
    Train randomly initialized model for num_steps.
    Compare loss curves.

    Returns: {
        "imputed_loss_curve": [float],
        "random_loss_curve": [float],
        "speedup_ratio": float,  # steps for random to reach imputed's final loss / num_steps
    }
    """
```

```python
def calibration_analysis(krige_variance, imputation_error):
    """
    For kriging method, compare predicted variance vs actual error.
    - Sort weights by krige variance into deciles
    - For each decile, compute mean actual error
    - Report Spearman correlation between variance and error

    Returns: {"spearman_rho": float, "decile_errors": [float]}
    """
```

```python
def full_validation_report(variants, ground_truth_model, device="cpu"):
    """
    Run all validation metrics on all imputed variants.
    Returns: dict with all results, ready for report generation.
    """
    data_iter = build_synthetic_data(256, 128, 10, device, dtype=torch.long)

    report = {}
    for name, (model, meta) in variants.items():
        report[name] = {
            "zero_shot_ppl": compute_perplexity(model, data_iter, device),
            "weight_distance": weight_distance(model, ground_truth_model),
            "convergence": convergence_speedup(model, ground_truth_model, device=device),
        }
        if meta.get("krige_var_k") is not None:
            error_k = meta["slot_k_pred"] - ground_truth_model.slot_k_emb.data
            report[name]["calibration"] = calibration_analysis(meta["krige_var_k"], error_k)

    return report
```

### 4.8 `src/slot_impute/report.py`

```python
def generate_report(validation_results, config):
    """
    Generates a markdown report with:
    1. Experiment configuration (table)
    2. Anchor training results (per-M perplexity table)
    3. Interferometry: signal ratio per M, example signal/noise slots
    4. Imputation quality: ppl comparison table, weight distance table
    5. Convergence speedup: loss curves (as numbers, or ASCII plot)
    6. Kriging calibration: scatter stats, decile table
    7. Cost analysis: training cost vs imputation cost
    8. Bonus M=768 extrapolation results (if run)
    9. Conclusions: did imputation work? is it worth scaling?

    Returns: markdown string, also writes to file.
    """
```

### 4.9 `experiments/m_variation/config.yaml`

```yaml
experiment:
  name: "slot_m_variation"
  description: "Test slot pool weight imputation across M dimension"

model:
  d_model: 512
  num_heads: 8
  num_rounds: 3
  alpha: 0.9
  gamma: 0.15

training:
  steps: 500
  lr: 0.0005
  seq_len: 128
  vocab_size: 256

anchors:
  # M values to train
  m_values: [128, 192, 256, 384, 512]
  seeds: [42, 123, 999]
  # M=256 is ground truth (not used as anchor)
  target_m: 256
  # Optional extrapolation target
  extrapolate_m: 768

imputation:
  interpolation_methods: ["linear", "spline", "krige"]
  variants:
    - name: "A_full"
      anchors: "all"        # use all anchor M values
      interpolation: "linear"
      signal_only: false
      non_slot: "average"
    - name: "B_boundary"
      anchors: [128, 512]   # boundary only
      interpolation: "linear"
      signal_only: false
      non_slot: "average"
    - name: "C_signal"
      anchors: "all"
      interpolation: "linear"
      signal_only: true
      non_slot: "average"
    - name: "D_naive"
      anchors: [128, 512]
      interpolation: "linear"
      signal_only: false
      non_slot: "best"

validation:
  pp_batches: 10
  convergence_steps: 100
  device: "cpu"  # or "mps" for M2 GPU

output:
  checkpoint_dir: "checkpoints/"
  report_path: "report.md"
```

---

## 5. Pipeline Scripts

### 5.1 `scripts/train_anchors.sh`

```bash
#!/bin/bash
# Train all anchor models. Reads config.yaml for M values and seeds.
# Usage: bash scripts/train_anchors.sh [--parallel N]

CONFIG="experiments/m_variation/config.yaml"
CHECKPOINT_DIR="checkpoints"
DEVICE="cpu"  # or "mps"

# Parse M values and seeds from config (or hardcode for MVP)
M_VALUES=(128 192 256 384 512)
SEEDS=(42 123 999)

for M in "${M_VALUES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Training M=$M seed=$SEED..."
        python -m slot_impute.model \
            --num-slots $M \
            --seed $SEED \
            --save-dir "$CHECKPOINT_DIR" \
            --device "$DEVICE"
    done
done

echo "Done. Checkpoints in $CHECKPOINT_DIR/"
```

### 5.2 `scripts/run_pipeline.sh`

```bash
#!/bin/bash
# Full pipeline: train -> interferometry -> variogram -> impute -> validate -> report
# Usage: bash scripts/run_pipeline.sh [--skip-train]

set -e

if [ "$1" != "--skip-train" ]; then
    bash scripts/train_anchors.sh
fi

CONFIG="experiments/m_variation/config.yaml"
CHECKPOINT_DIR="checkpoints"

echo "=== Phase 2: Interferometry ==="
python -m slot_impute.interferometry --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG"

echo "=== Phase 3: Variogram ==="
python -m slot_impute.variogram --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG"

echo "=== Phase 4: Imputation ==="
python -m slot_impute.impute --checkpoint-dir "$CHECKPOINT_DIR" --config "$CONFIG" --output-dir imputed/

echo "=== Phase 5: Validation ==="
python -m slot_impute.validate --checkpoint-dir "$CHECKPOINT_DIR" --imputed-dir imputed/ --config "$CONFIG"

echo "=== Generate Report ==="
python -m slot_impute.report --results validation_results.json --output report.md

echo "=== Done. See report.md ==="
```

---

## 6. Data Flow

```
Phase 1: Training
  config.yaml → model.py → checkpoints/M{128..512}_seed{42,123,999}.pt

Phase 2: Interferometry
  checkpoints/*.pt → interferometry.py → signal_masks/{128,192,384,512}.pt
                                         interferometry_stats.json

Phase 3: Variogram
  checkpoints/*.pt → variogram.py → variogram_models/{linear,spline,krige}.pt

Phase 4: Imputation
  checkpoints/*.pt + signal_masks/ + variogram_models/
    → impute.py → imputed/{A_full,B_boundary,C_signal,D_naive}.pt

Phase 5: Validation
  imputed/*.pt + checkpoints/M256_seed42.pt (ground truth)
    → validate.py → validation_results.json

Report:
  validation_results.json + interferometry_stats.json
    → report.py → report.md
```

---

## 7. Expected Outputs

### Quantitative

| Metric | Expected range | Interpretation if met |
|---|---|---|
| Anchor training perplexity (M=128) | 20-50 on synthetic | Training working |
| Anchor training perplexity (M=512) | 15-40 on synthetic | Larger M = better |
| Imputed zero-shot ppl / ground truth ppl | 0.8-2.0 | Imputation directionally correct |
| Slot pool cosine similarity (imputed vs GT) | 0.1-0.5 | Not random |
| Convergence speedup ratio | 1.5-4.0× | Warm start helps |
| Signal ratio per M | 0.1-0.5 of weights | Some weights are structurally constrained |
| Spearman ρ (variance vs error) | > 0.1 | Kriging uncertainty is informative |

### Qualitative

- Do slot weights shift smoothly with M, or chaotically?
- Does signal ratio decrease with M (more redundancy in larger pools)?
- Does boundary-only imputation (variant B) work nearly as well as all-4 (variant A)?
- Does signal-only fill (variant C) outperform full fill (variant A)?

---

## 8. Bonus: M=768 Extrapolation

After the main experiment, test extrapolation beyond the anchor range:

1. **Impute M=768** using variograms fit from M ≤ 512
2. **Fine-tune M=768** imputed for 100 steps
3. **Compare** against:
   - M=512 fully trained (500 steps)
   - M=256 fully trained (500 steps, ground truth)

Add to config:
```yaml
bonus:
  extrapolate_m: 768
  fine_tune_steps: 100
```

Hypothesis: A 768-slot model with an imputed slot pool and 100 fine-tuning steps
should match or exceed a fully-trained 512-slot model (500 steps), because the
extra 256 slots add capacity that brief fine-tuning can leverage. This is the
cost-savings thesis in miniature.

---

## 9. Implementation Order

1. **`reference.py`** — copy from murmurative-attention, verify imports work
2. **`model.py`** — MurmurativeProbe with checkpoint save/load, training loop
3. **`extract.py`** — weight extraction/injection utilities
4. **Phase 1**: train all 15 models (can start immediately once model.py works)
5. **`interferometry.py`** — Hungarian alignment, signal masks
6. **`variogram.py`** — linear interpolation (start simple)
7. **`impute.py`** — build imputed models
8. **`validate.py`** — metrics
9. **`report.py`** — markdown report
10. **Spline + kriging** interpolation (add after linear works)
11. **Bonus**: M=768 extrapolation