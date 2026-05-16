#!/usr/bin/env python3
"""Collect HyFormer ablation validation metrics and draw a compact figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional


VAL_RE = re.compile(r"Validation \| AUC: ([0-9.eE+-]+), LogLoss: ([0-9.eE+-]+|inf)")


def parse_log(path: Path) -> Dict[str, float]:
    best_auc = float("-inf")
    best_logloss = float("inf")
    last_auc = float("nan")
    last_logloss = float("nan")
    epochs = 0
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            match = VAL_RE.search(line)
            if not match:
                continue
            epochs += 1
            auc = float(match.group(1))
            logloss = float(match.group(2))
            last_auc = auc
            last_logloss = logloss
            if auc > best_auc:
                best_auc = auc
                best_logloss = logloss
    if best_auc == float("-inf"):
        best_auc = float("nan")
    return {
        "best_auc": best_auc,
        "best_logloss": best_logloss,
        "last_auc": last_auc,
        "last_logloss": last_logloss,
        "epochs": epochs,
    }


def load_variant_config(variant_dir: Path) -> Dict[str, object]:
    cfg_paths = sorted(variant_dir.glob("*.best_model/train_config.json"))
    if not cfg_paths:
        return {}
    try:
        return json.loads(cfg_paths[-1].read_text())
    except Exception:
        return {}


def write_svg(rows: List[Dict[str, object]], output_path: Path) -> None:
    rows = [r for r in rows if isinstance(r["best_auc"], float) and not math.isnan(r["best_auc"])]
    width = 1100
    bar_h = 28
    gap = 9
    left = 260
    right = 80
    top = 50
    height = top + len(rows) * (bar_h + gap) + 55
    min_auc = min((float(r["best_auc"]) for r in rows), default=0.0)
    max_auc = max((float(r["best_auc"]) for r in rows), default=1.0)
    floor = max(0.0, min_auc - 0.01)
    ceil = min(1.0, max_auc + 0.005)
    span = max(ceil - floor, 1e-6)
    plot_w = width - left - right

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-family="Arial" font-size="22" font-weight="700">HyFormer Ablation Validation AUC</text>',
    ]
    for idx, row in enumerate(rows):
        y = top + idx * (bar_h + gap)
        auc = float(row["best_auc"])
        name = str(row["variant"])
        bar_w = max(2.0, (auc - floor) / span * plot_w)
        color = "#2f6fbd" if idx == 0 else "#6b9fd8"
        parts.append(f'<text x="24" y="{y + 20}" font-family="Arial" font-size="14">{name}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.2f}" height="{bar_h}" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{left + bar_w + 8:.2f}" y="{y + 20}" font-family="Arial" font-size="14">{auc:.6f}</text>')
    parts.append(f'<text x="{left}" y="{height - 20}" font-family="Arial" font-size="12" fill="#555">axis floor={floor:.4f}, ceil={ceil:.4f}</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_png_if_available(rows: List[Dict[str, object]], output_path: Path) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        return str(exc)

    rows = [r for r in rows if isinstance(r["best_auc"], float) and not math.isnan(r["best_auc"])]
    labels = [str(r["variant"]) for r in rows]
    aucs = [float(r["best_auc"]) for r in rows]
    fig_h = max(4, 0.42 * len(rows) + 1.5)
    plt.figure(figsize=(11, fig_h))
    colors = ["#2f6fbd"] + ["#6b9fd8"] * max(0, len(rows) - 1)
    plt.barh(labels, aucs, color=colors)
    plt.gca().invert_yaxis()
    lo = max(0.0, min(aucs) - 0.01) if aucs else 0.0
    hi = min(1.0, max(aucs) + 0.005) if aucs else 1.0
    plt.xlim(lo, hi)
    plt.xlabel("Best validation AUC")
    plt.title("HyFormer Ablation Validation AUC")
    for i, auc in enumerate(aucs):
        plt.text(auc + (hi - lo) * 0.005, i, f"{auc:.6f}", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if variant_dir.name == "ablation_results" or variant_dir.name.startswith("ablation_best."):
            continue
        metrics = parse_log(variant_dir / "logs" / "train.log")
        cfg = load_variant_config(variant_dir / "ckpt")
        rows.append({
            "variant": variant_dir.name,
            **metrics,
            "seq_encoder_type": cfg.get("seq_encoder_type", ""),
            "rank_mixer_mode": cfg.get("rank_mixer_mode", ""),
            "query_context_mode": cfg.get("query_context_mode", ""),
            "use_time_buckets": cfg.get("use_time_buckets", ""),
            "use_rope": cfg.get("use_rope", ""),
            "num_queries": cfg.get("num_queries", ""),
            "num_hyformer_blocks": cfg.get("num_hyformer_blocks", ""),
        })

    rows.sort(key=lambda r: float(r["best_auc"]) if not math.isnan(float(r["best_auc"])) else -1.0, reverse=True)

    csv_path = out_dir / "ablation_results.csv"
    fieldnames = [
        "variant", "best_auc", "best_logloss", "last_auc", "last_logloss", "epochs",
        "seq_encoder_type", "rank_mixer_mode", "query_context_mode",
        "use_time_buckets", "use_rope", "num_queries", "num_hyformer_blocks",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_svg(rows, out_dir / "ablation_auc.svg")
    png_error = write_png_if_available(rows, out_dir / "ablation_auc.png")
    summary = {
        "best_variant": rows[0]["variant"] if rows else None,
        "best_auc": rows[0]["best_auc"] if rows else None,
        "csv": str(csv_path),
        "svg": str(out_dir / "ablation_auc.svg"),
        "png": str(out_dir / "ablation_auc.png") if png_error is None else None,
        "png_error": png_error,
    }
    (out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
