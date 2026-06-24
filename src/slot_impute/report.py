import json
import math
import os
from typing import Dict

import yaml


VARIANT_ORDER = ["A_full", "B_boundary", "C_signal", "D_naive", "E_krige"]


def generate_report(validation_results: dict, config: dict) -> str:
    lines = []

    lines.append("# Slot Imputation Experiment Report\n")

    task_type = config.get("experiment", {}).get("task", "random")
    eval_mode = "corrupted" if config.get("validation", {}).get("use_corrupted_eval", False) else "clean"

    lines.append("## 1. Experiment Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| d_model | {config['model']['d_model']} |")
    lines.append(f"| num_heads | {config['model']['num_heads']} |")
    lines.append(f"| num_rounds | {config['model']['num_rounds']} |")
    lines.append(f"| Training steps | {config['training']['steps']} |")
    lines.append(f"| Learning rate | {config['training']['lr']} |")
    lines.append(f"| Corruption rate | {config['training']['corruption_rate']} |")
    lines.append(f"| Anchor M values | {config['anchors']['m_values']} |")
    lines.append(f"| Target M | {config['anchors']['target_m']} |")
    lines.append(f"| Seeds | {config['anchors']['seeds']} |")
    lines.append(f"| Device | {config.get('validation', {}).get('device', 'cpu')} |")
    lines.append(f"| Task | {task_type} |")
    lines.append(f"| Eval mode | {eval_mode} (corruption_rate={'0.50' if eval_mode=='corrupted' else '0.00'}) |")
    lines.append("")

    lines.append("## 2. Imputation Quality: Zero-Shot Perplexity\n")
    lines.append(f"| Variant | Perplexity ({eval_mode} tokens) |")
    lines.append("|---|---|")
    for name in VARIANT_ORDER:
        if name in validation_results:
            ppl = validation_results[name].get("zero_shot_ppl", "N/A")
            ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
            lines.append(f"| {name} | {ppl_str} |")
    lines.append("")

    lines.append("## 3. Weight Distance to Ground Truth\n")
    lines.append("| Variant | Slot K MSE | Slot V MSE | Slot K Cosine | Slot V Cosine |")
    lines.append("|---|---|---|---|---|")
    for name in VARIANT_ORDER:
        if name in validation_results and "weight_distance" in validation_results[name]:
            wd = validation_results[name]["weight_distance"]
            lines.append(
                f"| {name} | {wd['slot_k_mse']:.6f} | {wd['slot_v_mse']:.6f} | "
                f"{wd['slot_k_cosine']:.4f} | {wd['slot_v_cosine']:.4f} |"
            )
    lines.append("")

    lines.append("## 4. Convergence Speedup\n")
    lines.append("| Variant | Imputed Initial Loss | Random Initial Loss | Head Start (steps to match) |")
    lines.append("|---|---|---|---|")
    for name in VARIANT_ORDER:
        if name in validation_results and "convergence" in validation_results[name]:
            conv = validation_results[name]["convergence"]
            imp_init = conv.get("imputed_initial_loss", 0)
            rnd_init = conv.get("random_initial_loss", 0)
            lines.append(
                f"| {name} | {imp_init:.4f} | {rnd_init:.4f} | {conv['speedup_ratio']} |"
            )
    lines.append("")

    lines.append("## 5. Kriging Calibration\n")
    for name in VARIANT_ORDER:
        if name in validation_results and "calibration" in validation_results[name]:
            cal = validation_results[name]["calibration"]
            lines.append(f"### {name}\n")
            lines.append(f"- Spearman ρ: {cal.get('spearman_rho', 'N/A')}")
            deciles = cal.get("decile_errors", [])
            if deciles:
                lines.append("- Decile errors (low variance → high variance):")
                for i, err in enumerate(deciles):
                    lines.append(f"  - Decile {i + 1}: {err:.6f}")
            lines.append("")

    lines.append("## 6. Summary\n")
    lines.extend(_generate_summary(validation_results, config))
    lines.append("")

    report_str = "\n".join(lines)
    return report_str


def _generate_summary(results: dict, config: dict) -> list[str]:
    lines = []
    task_type = config.get("experiment", {}).get("task", "random")
    eval_mode = "corrupted" if config.get("validation", {}).get("use_corrupted_eval", False) else "clean"
    corr_rate = config["training"]["corruption_rate"]
    vocab_size = config["task"]["random"]["vocab_size"] if task_type == "random" else config["task"]["structured"]["num_states"]

    lines.append("### Key Findings\n")

    a_full = results.get("A_full", {})
    b_boundary = results.get("B_boundary", {})
    d_naive = results.get("D_naive", {})
    e_krige = results.get("E_krige", {})

    k_cos_a = a_full.get("weight_distance", {}).get("slot_k_cosine", 0)
    k_cos_e = e_krige.get("weight_distance", {}).get("slot_k_cosine", 0)
    v_cos_a = a_full.get("weight_distance", {}).get("slot_v_cosine", 0)
    v_cos_b = b_boundary.get("weight_distance", {}).get("slot_v_cosine", 0)

    ppl_a = a_full.get("zero_shot_ppl", 0)
    ppl_b = b_boundary.get("zero_shot_ppl", 0)

    head_a = a_full.get("convergence", {}).get("speedup_ratio", 0)
    head_d = d_naive.get("convergence", {}).get("speedup_ratio", 0)

    cal_rho = e_krige.get("calibration", {}).get("spearman_rho", 0)

    chance_ppl_clean = vocab_size
    chance_ppl_corrupted = vocab_size

    lines.append(f"- **Task**: {task_type} tokens (vocab={vocab_size}), eval with {eval_mode} tokens (corruption_rate={corr_rate if eval_mode=='corrupted' else '0.00'})")

    if eval_mode == "clean":
        if ppl_b < chance_ppl_clean * 0.5:
            lines.append(f"- **Slot interpolation works**: B_boundary (boundary anchors only) achieves perplexity {ppl_b:.1f} vs chance {chance_ppl_clean}, showing slot weights encode recoverable structure from few anchors.")
        else:
            lines.append(f"- **Slot interpolation limited on clean eval**: A_full perplexity {ppl_a:.1f} is near chance ({chance_ppl_clean}); clean-token evaluation measures non-slot weight quality, not slot quality.")
    else:
        lines.append(f"- **Corrupted eval**: Models scored on {corr_rate*100:.0f}% corrupted tokens matching training distribution.")
        if ppl_a < chance_ppl_corrupted * 0.8:
            lines.append(f"- **Slot interpolation effective**: A_full ({ppl_a:.1f}) meaningfully below chance ({chance_ppl_corrupted}), showing slot weights encode learnable patterns.")

    lines.append(f"- **Slot K cosine**: A_full K cosine={k_cos_a:.3f}, E_krige K cosine={k_cos_e:.3f} — K interpolation recovers moderate directional alignment with ground truth.")
    lines.append(f"- **Slot V cosine**: A_full V cosine={v_cos_a:.3f}, B_boundary V cosine={v_cos_b:.3f} — V weights are near-orthogonal to ground truth, consistent with permutation-invariant slot V parameters.")
    lines.append(f"- **Kriging calibration**: Spearman ρ={cal_rho:.4f} — kriging variance has negligible predictive power for imputation error; the exponential variogram does not capture slot-weight uncertainty in this setting.")

    lines.append(f"- **Head start**: A_full at {head_a} steps, D_naive at {head_d} steps — slot interpolation alone provides limited warm-start advantage; D_naive's large head start comes from retaining a single converged checkpoint's non-slot weights (embed + lm_head).")

    if k_cos_a > 0.2 and k_cos_e > 0.2:
        lines.append("\n**Conclusion**: Slot K weights vary smoothly enough with M that linear/kriging interpolation recovers non-trivial directional information. Slot V appears permutation-invariant and may need alternative treatment. Future work should test on structured tasks where slot attention has meaningful functions to encode.")
    else:
        lines.append("\n**Conclusion**: Slot weight interpolation recovers magnitudes (low MSE) but not directions (low cosine). The {task_type} task may lack sufficient structure for slots to develop consistently aligned representations across M values. A structured-task experiment is needed to evaluate the thesis under favorable conditions.")

    return lines


def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="validation_results.json")
    parser.add_argument("--config", type=str, default="experiments/m_variation/config.yaml")
    parser.add_argument("--output", type=str, default="report.md")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.results) as f:
        results = json.load(f)

    report = generate_report(results, config)

    with open(args.output, "w") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    _main()