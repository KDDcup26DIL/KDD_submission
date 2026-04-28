#!/usr/bin/env python3
"""Save a checkpoint architecture summary.

The script rebuilds the model from ``train_config.json`` and ``schema.json``
without opening parquet data. It writes:

- ``architecture.txt``: readable module tree and parameter counts
- ``architecture.json``: structured summary
- ``architecture.dot``: Graphviz module tree
- ``architecture.png``: rendered from dot when the ``dot`` executable exists
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


NUM_TIME_BUCKETS = 65

BASE_FALLBACK_MODEL_CFG: Dict[str, Any] = {
    "d_model": 64,
    "emb_dim": 64,
    "num_queries": 1,
    "num_heads": 4,
    "seq_encoder_type": "transformer",
    "hidden_mult": 4,
    "dropout_rate": 0.01,
    "seq_top_k": 50,
    "seq_causal": False,
    "action_num": 1,
    "num_time_buckets": NUM_TIME_BUCKETS,
    "rank_mixer_mode": "full",
    "use_rope": False,
    "rope_base": 10000.0,
    "emb_skip_threshold": 0,
    "seq_id_threshold": 10000,
    "ns_tokenizer_type": "rankmixer",
    "user_ns_tokens": 0,
    "item_ns_tokens": 0,
}


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    checkpoint_prefix: str
    code_dir_parts: Tuple[str, ...]
    class_name: str
    layer_key: str
    default_layers: int

    @property
    def fallback_model_cfg(self) -> Dict[str, Any]:
        cfg = dict(BASE_FALLBACK_MODEL_CFG)
        cfg[self.layer_key] = self.default_layers
        return cfg

    @property
    def model_cfg_keys(self) -> List[str]:
        return list(self.fallback_model_cfg.keys())


MODEL_SPECS: Dict[str, ModelSpec] = {
    "dcnv2": ModelSpec(
        display_name="PCVRDCNv2",
        checkpoint_prefix="dcnv2",
        code_dir_parts=("dcnv2", "dcnv2_submission", "dcnv2_evaluation"),
        class_name="PCVRDCNv2",
        layer_key="num_dcnv2_layers",
        default_layers=2,
    ),
    "hyformer": ModelSpec(
        display_name="PCVRHyFormer",
        checkpoint_prefix="hyformer",
        code_dir_parts=("hyformer", "hyformer_submission", "hyformer_evaluation"),
        class_name="PCVRHyFormer",
        layer_key="num_hyformer_blocks",
        default_layers=2,
    ),
    "wukong": ModelSpec(
        display_name="PCVRWuKong",
        checkpoint_prefix="Wukong",
        code_dir_parts=("Wukong", "Wukong_submission", "Wukong_evaluation"),
        class_name="PCVRWuKong",
        layer_key="num_wukong_layers",
        default_layers=3,
    ),
}


@dataclass
class FeatureSchema:
    entries: List[Tuple[int, int, int]]
    total_dim: int

    @classmethod
    def from_columns(cls, columns: Iterable[Iterable[int]]) -> "FeatureSchema":
        entries: List[Tuple[int, int, int]] = []
        offset = 0
        for col in columns:
            fid = int(col[0])
            dim = int(col[-1])
            entries.append((fid, offset, dim))
            offset += dim
        return cls(entries=entries, total_dim=offset)


@dataclass
class SchemaInfo:
    user_int_schema: FeatureSchema
    item_int_schema: FeatureSchema
    user_dense_schema: FeatureSchema
    item_dense_schema: FeatureSchema
    user_int_vocab_sizes: List[int]
    item_int_vocab_sizes: List[int]
    seq_domain_vocab_sizes: Dict[str, List[int]]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_model_name(value: str) -> str:
    lowered = value.lower()
    for key, spec in MODEL_SPECS.items():
        if key in lowered or spec.checkpoint_prefix.lower() in lowered:
            return key
    supported = ", ".join(spec.checkpoint_prefix for spec in MODEL_SPECS.values())
    raise ValueError(f"Cannot infer model type from {value!r}. Supported names: {supported}")


def infer_model_key(model_dir: Path) -> str:
    for part in reversed(model_dir.parts):
        try:
            return normalize_model_name(part)
        except ValueError:
            continue
    return normalize_model_name(str(model_dir))


def find_latest_model_dir(repo_root: Path, model_name: str) -> Tuple[Path, str]:
    model_key = normalize_model_name(model_name)
    spec = MODEL_SPECS[model_key]
    checkpoint_root = repo_root / "checkpoint"
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_root}")

    run_dirs = [
        path for path in checkpoint_root.iterdir()
        if path.is_dir() and path.name.lower().startswith(spec.checkpoint_prefix.lower())
    ]
    candidates: List[Path] = []
    for run_dir in run_dirs:
        candidates.extend(
            path for path in run_dir.iterdir()
            if path.is_dir() and path.name.endswith(".best_model")
        )

    if not candidates:
        raise FileNotFoundError(
            f"No best_model checkpoint found for {spec.checkpoint_prefix!r} under {checkpoint_root}"
        )

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0], model_key


def resolve_model_target(repo_root: Path, value: str) -> Tuple[Path, str]:
    path = Path(value).expanduser()
    if path.is_dir():
        resolved = path.resolve()
        has_checkpoint_sidecar = (
            (resolved / "schema.json").is_file()
            or (resolved / "train_config.json").is_file()
        )
        if has_checkpoint_sidecar:
            return resolved, infer_model_key(resolved)
        try:
            return find_latest_model_dir(repo_root, value)
        except ValueError:
            return resolved, infer_model_key(resolved)
    return find_latest_model_dir(repo_root, value)


def resolve_model_cfg(spec: ModelSpec, train_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    fallback_model_cfg = spec.fallback_model_cfg
    for key in spec.model_cfg_keys:
        if key == "num_time_buckets":
            if "num_time_buckets" in train_config:
                cfg[key] = train_config["num_time_buckets"]
            elif "use_time_buckets" in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config["use_time_buckets"] else 0
            else:
                cfg[key] = fallback_model_cfg[key]
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        elif key == spec.layer_key and "num_hyformer_blocks" in train_config:
            cfg[key] = train_config["num_hyformer_blocks"]
        else:
            cfg[key] = fallback_model_cfg[key]
    return cfg


def parse_schema(schema_path: Path) -> SchemaInfo:
    raw = load_json(schema_path)

    user_int_schema = FeatureSchema.from_columns(raw["user_int"])
    item_int_schema = FeatureSchema.from_columns(raw["item_int"])
    user_dense_schema = FeatureSchema.from_columns(raw["user_dense"])
    item_dense_schema = FeatureSchema(entries=[], total_dim=0)

    user_int_vocab_sizes: List[int] = []
    for _, vocab_size, dim in raw["user_int"]:
        user_int_vocab_sizes.extend([int(vocab_size)] * int(dim))

    item_int_vocab_sizes: List[int] = []
    for _, vocab_size, dim in raw["item_int"]:
        item_int_vocab_sizes.extend([int(vocab_size)] * int(dim))

    seq_domain_vocab_sizes: Dict[str, List[int]] = {}
    for domain in sorted(raw["seq"].keys()):
        domain_cfg = raw["seq"][domain]
        ts_fid = domain_cfg.get("ts_fid")
        seq_domain_vocab_sizes[domain] = [
            int(vocab_size)
            for fid, vocab_size in domain_cfg["features"]
            if fid != ts_fid
        ]

    return SchemaInfo(
        user_int_schema=user_int_schema,
        item_int_schema=item_int_schema,
        user_dense_schema=user_dense_schema,
        item_dense_schema=item_dense_schema,
        user_int_vocab_sizes=user_int_vocab_sizes,
        item_int_vocab_sizes=item_int_vocab_sizes,
        seq_domain_vocab_sizes=seq_domain_vocab_sizes,
    )


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for _, offset, length in schema.entries:
        vocab_size = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vocab_size, offset, length))
    return specs


def resolve_ns_groups(
    schema_info: SchemaInfo,
    ns_groups_path: Optional[Path],
) -> Tuple[List[List[int]], List[List[int]]]:
    if not ns_groups_path or not ns_groups_path.exists():
        user_groups = [[i] for i in range(len(schema_info.user_int_schema.entries))]
        item_groups = [[i] for i in range(len(schema_info.item_int_schema.entries))]
        return user_groups, item_groups

    ns_groups_cfg = load_json(ns_groups_path)
    user_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(schema_info.user_int_schema.entries)
    }
    item_fid_to_idx = {
        fid: i for i, (fid, _, _) in enumerate(schema_info.item_int_schema.entries)
    }
    user_groups = [
        [user_fid_to_idx[int(fid)] for fid in fids]
        for fids in ns_groups_cfg["user_ns_groups"].values()
    ]
    item_groups = [
        [item_fid_to_idx[int(fid)] for fid in fids]
        for fids in ns_groups_cfg["item_ns_groups"].values()
    ]
    return user_groups, item_groups


def format_count(value: int) -> str:
    return f"{value:,}"


def direct_param_count(module: Any) -> int:
    return sum(param.numel() for param in module.parameters(recurse=False))


def total_param_count(module: Any) -> int:
    return sum(param.numel() for param in module.parameters())


def trainable_param_count(module: Any) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def module_rows(model: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        path = name or "<root>"
        rows.append({
            "path": path,
            "class": module.__class__.__name__,
            "depth": 0 if not name else name.count(".") + 1,
            "direct_params": direct_param_count(module),
            "total_params": total_param_count(module),
        })
    return rows


def write_text_summary(
    path: Path,
    spec: ModelSpec,
    model: Any,
    model_dir: Path,
    schema_path: Path,
    train_config_path: Optional[Path],
    model_cfg: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    lines = [
        f"{spec.display_name} Architecture",
        "=" * (len(spec.display_name) + 13),
        f"model_dir: {model_dir}",
        f"schema_json: {schema_path}",
        f"train_config_json: {train_config_path if train_config_path else 'missing'}",
        f"total_params: {format_count(total_param_count(model))}",
        f"trainable_params: {format_count(trainable_param_count(model))}",
        "",
        "Resolved model config:",
    ]
    for key in spec.model_cfg_keys:
        lines.append(f"  {key}: {model_cfg[key]}")

    lines.extend(["", "Module tree:"])
    for row in rows:
        indent = "  " * row["depth"]
        direct = row["direct_params"]
        total = row["total_params"]
        param_text = f" direct={format_count(direct)} total={format_count(total)}"
        lines.append(f"{indent}- {row['path']} [{row['class']}]{param_text}")

    lines.extend(["", "Parameters:"])
    for name, param in model.named_parameters():
        lines.append(
            f"  {name}: shape={list(param.shape)} "
            f"numel={format_count(param.numel())} requires_grad={param.requires_grad}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dot_id(path: str) -> str:
    safe = path.replace("<", "").replace(">", "").replace(".", "_")
    return "root" if safe == "root" else safe


def dot_label(row: Dict[str, Any]) -> str:
    return (
        f"{row['path']}\\n{row['class']}\\n"
        f"direct={format_count(row['direct_params'])}"
    )


def write_dot(path: Path, spec: ModelSpec, rows: List[Dict[str, Any]]) -> None:
    by_path = {row["path"]: row for row in rows}
    lines = [
        f"digraph {spec.display_name} {{",
        "  graph [rankdir=TB, bgcolor=\"white\"];",
        "  node [shape=box, style=\"rounded,filled\", fillcolor=\"#f7f9fb\", color=\"#6b7280\", fontname=\"Arial\"];",
        "  edge [color=\"#9ca3af\"];",
    ]
    for row in rows:
        node_id = dot_id(row["path"])
        lines.append(f"  {node_id} [label=\"{dot_label(row)}\"];")

    for row in rows:
        path_name = row["path"]
        if path_name == "<root>":
            continue
        parent = "<root>" if "." not in path_name else path_name.rsplit(".", 1)[0]
        if parent in by_path:
            lines.append(f"  {dot_id(parent)} -> {dot_id(path_name)};")

    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_render_png(dot_path: Path, png_path: Path) -> bool:
    if shutil.which("dot") is None:
        return False
    subprocess.run(
        ["dot", "-Tpng", str(dot_path), "-o", str(png_path)],
        check=True,
    )
    return True


def build_model(
    spec: ModelSpec,
    code_dir: Path,
    schema_info: SchemaInfo,
    model_cfg: Dict[str, Any],
    ns_groups_path: Optional[Path],
) -> Any:
    sys.path.insert(0, str(code_dir))
    model_module = __import__("model")
    model_cls = getattr(model_module, spec.class_name)

    user_ns_groups, item_ns_groups = resolve_ns_groups(schema_info, ns_groups_path)
    user_int_feature_specs = build_feature_specs(
        schema_info.user_int_schema,
        schema_info.user_int_vocab_sizes,
    )
    item_int_feature_specs = build_feature_specs(
        schema_info.item_int_schema,
        schema_info.item_int_vocab_sizes,
    )

    return model_cls(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=schema_info.user_dense_schema.total_dim,
        item_dense_dim=schema_info.item_dense_schema.total_dim,
        seq_vocab_sizes=schema_info.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Write architecture visualization files for a checkpoint directory or model name.",
    )
    parser.add_argument(
        "model",
        help=(
            "Checkpoint/model directory, or a model name such as dcnv2, hyformer, or Wukong. "
            "A model name resolves to the latest checkpoint/<name>_*/*.best_model directory."
        ),
    )
    parser.add_argument("--schema", default=None, help="schema.json path. Defaults to MODEL_DIR/schema.json.")
    parser.add_argument("--code-dir", default=None, help="Directory containing model.py. Defaults from model type.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to MODEL_DIR.")
    parser.add_argument("--prefix", default="architecture", help="Output filename prefix.")
    parser.add_argument("--no-png", action="store_true", help="Skip Graphviz PNG rendering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    model_dir, model_key = resolve_model_target(repo_root, args.model)
    spec = MODEL_SPECS[model_key]
    default_code_dir = repo_root.joinpath(*spec.code_dir_parts)
    code_dir = Path(args.code_dir).expanduser().resolve() if args.code_dir else default_code_dir
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else model_dir
    schema_path = Path(args.schema).expanduser().resolve() if args.schema else model_dir / "schema.json"
    train_config_path = model_dir / "train_config.json"

    if not model_dir.is_dir():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")
    if not code_dir.is_dir():
        raise FileNotFoundError(f"code_dir does not exist: {code_dir}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"schema.json does not exist: {schema_path}")

    print(f"model_type: {model_key}")
    print(f"model_dir: {model_dir}")
    print(f"code_dir: {code_dir}")

    train_config = load_json(train_config_path) if train_config_path.is_file() else {}
    model_cfg = resolve_model_cfg(spec, train_config)

    ns_groups_path: Optional[Path] = None
    ns_groups_value = train_config.get("ns_groups_json")
    if ns_groups_value:
        candidate = model_dir / Path(ns_groups_value).name
        ns_groups_path = candidate if candidate.exists() else Path(ns_groups_value).expanduser()

    schema_info = parse_schema(schema_path)
    model = build_model(spec, code_dir, schema_info, model_cfg, ns_groups_path)
    rows = module_rows(model)

    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / f"{args.prefix}.txt"
    json_path = output_dir / f"{args.prefix}.json"
    dot_path = output_dir / f"{args.prefix}.dot"
    png_path = output_dir / f"{args.prefix}.png"

    write_text_summary(
        txt_path,
        spec,
        model,
        model_dir,
        schema_path,
        train_config_path if train_config_path.is_file() else None,
        model_cfg,
        rows,
    )
    json_path.write_text(
        json.dumps({
            "model_type": model_key,
            "model_class": spec.class_name,
            "model_dir": str(model_dir),
            "schema_json": str(schema_path),
            "train_config_json": str(train_config_path) if train_config_path.is_file() else None,
            "model_cfg": model_cfg,
            "total_params": total_param_count(model),
            "trainable_params": trainable_param_count(model),
            "modules": rows,
            "parameters": [
                {
                    "name": name,
                    "shape": list(param.shape),
                    "numel": param.numel(),
                    "requires_grad": param.requires_grad,
                }
                for name, param in model.named_parameters()
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_dot(dot_path, spec, rows)

    rendered_png = False
    if not args.no_png:
        rendered_png = maybe_render_png(dot_path, png_path)

    print(f"saved: {txt_path}")
    print(f"saved: {json_path}")
    print(f"saved: {dot_path}")
    if rendered_png:
        print(f"saved: {png_path}")
    else:
        print("png: skipped (install graphviz dot, or omit --no-png to render when available)")


if __name__ == "__main__":
    main()
