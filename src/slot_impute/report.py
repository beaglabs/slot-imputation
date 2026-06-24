import json
import os
from typing import Dict

import yaml


def generate_report(validation_results: dict, config: dict) -> str:
    lines = []

    lines.append("# Slot Imputation Experiment Report\n")

    lines.append("## 1. Experiment Configuration\n")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| d_model | {config['model']['d_model']} |")
    lines.append(f"| num_heads | {config['model']['num_heads']} |")
    lines.append(f"| num_rounds | {config['model']['num_rounds']} |")
    lines.append(f"| Training steps | {config['training']['steps']} |")
    lines.append(f"| Learning rate | {config['training']['lr']} |")
    lines.append(f"| Anchor M values | {config['anchors']['m_values']} |")
    lines.append(f"| Target M | {config['anchors']['target_m']} |")
    lines.append(f"| Seeds | {config['anchors']['seeds']} |")
    lines.append(f"| Device | {config.get('validation', {}).get('device', 'cpu')} |")
    lines.append("")

    lines.append("## 2. Imputation Quality: Zero-Shot Perplexity\n")
    lines.append("| Variant | Perplexity |")
    lines.append("|---|---|")
    for name in ["A_full", "B_boundary", "C_signal", "D_naive"]:
        if name in validation_results:
            ppl = validation_results[name].get("zero_shot_ppl", "N/A")
            ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
            lines.append(f"| {name} | {ppl_str} |")
    lines.append("")

    lines.append("## 3. Weight Distance to Ground Truth\n")
    lines.append("| Variant | Slot K MSE | Slot V MSE | Slot K Cosine | Slot V Cosine |")
    lines.append("|---|---|---|---|---|")
    for name in ["A_full", "B_boundary", "C_signal", "D_naive"]:
        if name in validation_results and "weight_distance" in validation_results[name]:
            wd = validation_results[name]["weight_distance"]
            lines.append(
                f"| {name} | {wd['slot_k_mse']:.6f} | {wd['slot_v_mse']:.6f} | "
                f"{wd['slot_k_cosine']:.4f} | {wd['slot_v_cosine']:.4f} |"
            )
    lines.append("")

    lines.append("## 4. Convergence Speedup\n")
    lines.append("| Variant | Imputed Final Loss | Random Final Loss | Speedup Ratio |")
    lines.append("|---|---|---|---|")
    for name in ["A_full", "B_boundary", "C_signal", "D_naive"]:
        if name in validation_results and "convergence" in validation_results[name]:
            conv = validation_results[name]["convergence"]
            imp_final = conv["imputed_loss_curve"][-1] if conv["imputed_loss_curve"] else 0
            rnd_final = conv["random_loss_curve"][-1] if conv["random_loss_curve"] else 0
            lines.append(
                f"| {name} | {imp_final:.4f} | {rnd_final:.4f} | {conv['speedup_ratio']:.2f}x |"
            )
    lines.append("")

    lines.append("## 5. Kriging Calibration\n")
    for name in ["A_full", "B_boundary", "C_signal", "D_naive"]:
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

    lines.append("## 6. Qualitative Observations\n")
    lines.append("- *Add observations after running the experiment.*")
    lines.append("")

    lines.append("## 7. Conclusions\n")
    lines.append("- *Add conclusions after reviewing results.*")
    lines.append("")

    report_str = "\n".join(lines)
    return report_str


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