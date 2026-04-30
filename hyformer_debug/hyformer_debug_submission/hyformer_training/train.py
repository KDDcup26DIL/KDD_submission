"""PCVRHyFormer training entry point (self-contained baseline).

Usage:
    python train.py [--num_epochs 10] [--batch_size 256] ...

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""

import os
import json
import argparse
import logging
import glob
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from utils import set_seed, EarlyStopping, create_logger
from dataset import FeatureSchema, get_pcvr_data, NUM_TIME_BUCKETS
from model import PCVRHyFormer
from trainer import PCVRHyFormerRankingTrainer


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    """Build feature_specs of the form ``[(vocab_size, offset, length), ...]``
    ordered by the positions recorded in ``schema.entries``.
    """
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _summary(values: List[int]) -> str:
    if not values:
        return "count=0"
    return (
        f"count={len(values)}, min={min(values)}, max={max(values)}, "
        f"unique={len(set(values))}"
    )


def _log_data_profile(
    data_dir: str,
    schema_path: str,
    max_profile_files: int = 5,
    max_values_per_dense_col: int = 200000,
) -> None:
    """Log raw parquet/schema statistics for server/local data sanity checks.

    The dense-value scan is intentionally sampled to avoid delaying server
    training on large datasets. Metadata such as file count, row count, and row
    groups is still computed over all parquet files.
    """
    try:
        import numpy as np
        import pyarrow.parquet as pq

        logging.info("=== Data value profile begin ===")
        logging.info(f"profile data_dir={data_dir}, exists={os.path.exists(data_dir)}")
        logging.info(f"profile schema_path={schema_path}, exists={os.path.exists(schema_path)}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_cfg = json.load(f)

        user_int = schema_cfg.get("user_int", [])
        item_int = schema_cfg.get("item_int", [])
        user_dense = schema_cfg.get("user_dense", [])
        seq_cfg = schema_cfg.get("seq", {})
        logging.info(
            "profile schema counts: "
            f"user_int={len(user_int)}, item_int={len(item_int)}, "
            f"user_dense={len(user_dense)}, seq_domains={list(seq_cfg.keys())}"
        )
        logging.info(f"profile user_int vocab summary: {_summary([int(x[1]) for x in user_int])}")
        logging.info(f"profile item_int vocab summary: {_summary([int(x[1]) for x in item_int])}")
        logging.info(
            "profile user_dense dims: "
            + str([(int(fid), int(dim)) for fid, dim in user_dense])
        )
        for domain, cfg in seq_cfg.items():
            features = cfg.get("features", [])
            logging.info(
                f"profile seq[{domain}]: prefix={cfg.get('prefix')}, "
                f"ts_fid={cfg.get('ts_fid')}, features={len(features)}, "
                f"vocab_summary={_summary([int(x[1]) for x in features])}"
            )

        if os.path.isdir(data_dir):
            parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
            entries = sorted(os.listdir(data_dir))[:30]
            logging.info(f"profile data_dir first entries (30 shown): {entries}")
        else:
            parquet_files = [data_dir]
        if not parquet_files:
            logging.info("profile parquet file count=0")
            logging.info("=== Data value profile end ===")
            return

        total_rows = 0
        total_row_groups = 0
        file_rows = []
        row_group_rows = []
        first_pf = None
        for path in parquet_files:
            pf = pq.ParquetFile(path)
            if first_pf is None:
                first_pf = pf
            rows = pf.metadata.num_rows
            rgs = pf.metadata.num_row_groups
            total_rows += rows
            total_row_groups += rgs
            file_rows.append(rows)
            for rg in range(rgs):
                row_group_rows.append(pf.metadata.row_group(rg).num_rows)
        logging.info(
            f"profile parquet files={len(parquet_files)}, "
            f"total_rows={total_rows}, total_row_groups={total_row_groups}"
        )
        logging.info(
            "profile file rows summary: "
            f"min={min(file_rows)}, p50={int(np.percentile(file_rows, 50))}, "
            f"p90={int(np.percentile(file_rows, 90))}, max={max(file_rows)}"
        )
        logging.info(
            "profile row_group rows summary: "
            f"count={len(row_group_rows)}, min={min(row_group_rows)}, "
            f"p50={int(np.percentile(row_group_rows, 50))}, "
            f"p90={int(np.percentile(row_group_rows, 90))}, "
            f"p99={int(np.percentile(row_group_rows, 99))}, max={max(row_group_rows)}"
        )

        first_path = parquet_files[0]
        first_schema = first_pf.schema_arrow
        logging.info(
            f"profile first parquet: name={os.path.basename(first_path)}, "
            f"rows={first_pf.metadata.num_rows}, "
            f"row_groups={first_pf.metadata.num_row_groups}, "
            f"columns={len(first_schema.names)}"
        )
        logging.info(f"profile first parquet columns: {first_schema.names}")
        logging.info(
            "profile first parquet dense types: "
            + str([
                (name, str(first_schema.field(name).type))
                for name in first_schema.names
                if name.startswith("user_dense_feats_")
            ])
        )

        sample_files = parquet_files[:max_profile_files]
        label_counts: Dict[int, int] = {}
        dense_cols = [f"user_dense_feats_{int(fid)}" for fid, _ in user_dense]
        dense_values: Dict[str, List[float]] = {c: [] for c in dense_cols}
        dense_lens: Dict[str, List[int]] = {c: [] for c in dense_cols}
        dense_null_lists: Dict[str, int] = {c: 0 for c in dense_cols}
        dense_null_values: Dict[str, int] = {c: 0 for c in dense_cols}
        dense_total_values: Dict[str, int] = {c: 0 for c in dense_cols}

        columns = ["label_type"] + dense_cols
        for path in sample_files:
            available = set(pq.ParquetFile(path).schema_arrow.names)
            read_cols = [c for c in columns if c in available]
            table = pq.read_table(path, columns=read_cols)
            if "label_type" in table.column_names:
                for value in table["label_type"].to_pylist():
                    if value is None:
                        continue
                    label_counts[int(value)] = label_counts.get(int(value), 0) + 1
            for col_name in dense_cols:
                if col_name not in table.column_names:
                    continue
                for row in table[col_name].to_pylist():
                    if row is None:
                        dense_null_lists[col_name] += 1
                        continue
                    dense_lens[col_name].append(len(row))
                    for value in row:
                        if value is None:
                            dense_null_values[col_name] += 1
                            continue
                        dense_total_values[col_name] += 1
                        if len(dense_values[col_name]) < max_values_per_dense_col:
                            dense_values[col_name].append(float(value))

        logging.info(
            f"profile sampled files={len(sample_files)}, "
            f"sampled label_type counts={dict(sorted(label_counts.items()))}"
        )
        for col_name in dense_cols:
            values = np.asarray(dense_values[col_name], dtype=np.float64)
            lens = np.asarray(dense_lens[col_name], dtype=np.float64)
            if values.size == 0:
                logging.info(
                    f"profile dense {col_name}: no sampled values, "
                    f"null_lists={dense_null_lists[col_name]}, "
                    f"null_values={dense_null_values[col_name]}"
                )
                continue
            finite = values[np.isfinite(values)]
            nan_count = int(np.isnan(values).sum())
            inf_count = int(np.isinf(values).sum())
            zero_ratio = float((values == 0).mean())
            integer_ratio = float((np.abs(values - np.round(values)) < 1e-6).mean())
            if finite.size:
                q = np.percentile(finite, [0, 1, 25, 50, 75, 90, 99, 100])
                mean = float(finite.mean())
                std = float(finite.std())
                absmax = float(np.max(np.abs(finite)))
                stats = (
                    f"min={q[0]:.6g}, p1={q[1]:.6g}, p25={q[2]:.6g}, "
                    f"p50={q[3]:.6g}, p75={q[4]:.6g}, p90={q[5]:.6g}, "
                    f"p99={q[6]:.6g}, max={q[7]:.6g}, "
                    f"mean={mean:.6g}, std={std:.6g}, absmax={absmax:.6g}"
                )
            else:
                stats = "no finite values"
            len_stats = "no lengths"
            if lens.size:
                len_q = np.percentile(lens, [0, 50, 90, 99, 100])
                len_stats = (
                    f"len_min={len_q[0]:.0f}, len_p50={len_q[1]:.0f}, "
                    f"len_p90={len_q[2]:.0f}, len_p99={len_q[3]:.0f}, "
                    f"len_max={len_q[4]:.0f}"
                )
            logging.info(
                f"profile dense {col_name}: sampled_values={values.size}/"
                f"{dense_total_values[col_name]}, {stats}, "
                f"nan={nan_count}, inf={inf_count}, zero_ratio={zero_ratio:.6f}, "
                f"integer_like_ratio={integer_ratio:.6f}, "
                f"null_lists={dense_null_lists[col_name]}, "
                f"null_values={dense_null_values[col_name]}, {len_stats}"
            )
        logging.info("=== Data value profile end ===")
    except Exception as exc:
        logging.exception(f"Data value profile failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCVRHyFormer Training")

    # Paths (environment variables take precedence).
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Training data directory (env: TRAIN_DATA_PATH)')
    parser.add_argument('--schema_path', type=str, default=None,
                        help='Schema JSON path (defaults to <data_dir>/schema.json)')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint output directory (env: TRAIN_CKPT_PATH)')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory (env: TRAIN_LOG_PATH)')

    # Training hyperparameters.
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for both training and validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for dense parameters (AdamW)')
    parser.add_argument('--num_epochs', type=int, default=999,
                        help='Maximum number of training epochs '
                             '(typically terminated earlier by early stopping)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early-stopping patience '
                             '(number of validations without improvement)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Training device, e.g. cuda or cpu')

    # Data pipeline.
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of DataLoader workers')
    parser.add_argument('--buffer_batches', type=int, default=20,
                        help='Shuffle buffer size, in units of batches. '
                             'Lower values reduce memory usage.')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Fraction of training Row Groups to use (takes the first N%)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Fraction of all Row Groups used for validation (takes the tail)')
    parser.add_argument('--eval_every_n_steps', type=int, default=0,
                        help='Run validation every N steps '
                             '(0 = only at the end of each epoch)')
    parser.add_argument('--seq_max_lens', type=str,
                        default='seq_a:256,seq_b:256,seq_c:512,seq_d:512',
                        help='Per-domain sequence truncation, format: seq_d:256,seq_c:128')

    # Model hyperparameters.
    parser.add_argument('--d_model', type=int, default=64,
                        help='Backbone hidden dimension (output size of each block)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='Per-Embedding-table dimension (before projection)')
    parser.add_argument('--num_queries', type=int, default=1,
                        help='Number of Query tokens generated independently per sequence domain')
    parser.add_argument('--num_hyformer_blocks', type=int, default=2,
                        help='Number of stacked MultiSeqHyFormerBlock layers')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads (must satisfy d_model %% num_heads == 0)')
    parser.add_argument('--seq_encoder_type', type=str, default='transformer',
                        choices=['swiglu', 'transformer', 'longer'],
                        help='Sequence encoder variant: '
                             'swiglu = SwiGLU without attention, '
                             'transformer = standard self-attention, '
                             'longer = Top-K compressed encoder '
                             '(only this variant consumes --seq_top_k / --seq_causal)')
    parser.add_argument('--hidden_mult', type=int, default=4,
                        help='FFN inner-dim multiplier relative to d_model')
    parser.add_argument('--dropout_rate', type=float, default=0.01,
                        help='Dropout rate for the backbone '
                             '(seq id-embedding dropout is twice this value)')
    parser.add_argument('--seq_top_k', type=int, default=50,
                        help='Number of most-recent tokens kept by LongerEncoder '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--seq_causal', action='store_true', default=False,
                        help='Whether the LongerEncoder self-attention uses a causal mask '
                             '(only effective when --seq_encoder_type=longer)')
    parser.add_argument('--action_num', type=int, default=1,
                        help='Classifier output dimension '
                             '(1 = single binary-classification logit; >1 = multi-label)')
    parser.add_argument('--use_time_buckets', action='store_true', default=True,
                        help='Enable the time-bucket embedding (default on). '
                             'The actual bucket count is uniquely determined by '
                             'dataset.BUCKET_BOUNDARIES; this flag is a pure on/off switch.')
    parser.add_argument('--no_time_buckets', dest='use_time_buckets', action='store_false',
                        help='Disable the time-bucket embedding')
    parser.add_argument('--rank_mixer_mode', type=str, default='full',
                        choices=['full', 'ffn_only', 'none'],
                        help='RankMixerBlock mode: '
                             'full = token mixing + per-token FFN (requires d_model divisible by T), '
                             'ffn_only = per-token FFN only, '
                             'none = identity passthrough')
    parser.add_argument('--use_rope', action='store_true', default=False,
                        help='Enable RoPE positional encoding in sequence attention')
    parser.add_argument('--rope_base', type=float, default=10000.0,
                        help='RoPE base frequency (default 10000)')

    # Loss function.
    parser.add_argument('--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Loss type: bce = BCEWithLogits, focal = Focal Loss')
    parser.add_argument('--focal_alpha', type=float, default=0.1,
                        help='Focal Loss positive-class weight alpha '
                             '(effective only when --loss_type=focal)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal Loss focusing parameter gamma '
                             '(effective only when --loss_type=focal)')

    # Sparse optimizer.
    parser.add_argument('--sparse_lr', type=float, default=0.05,
                        help='Learning rate for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--sparse_weight_decay', type=float, default=0.0,
                        help='Weight decay for sparse parameters (Adagrad over Embeddings)')
    parser.add_argument('--reinit_sparse_after_epoch', type=int, default=1,
                        help='Starting from the N-th epoch, at the end of every epoch '
                             're-initialize Embeddings with vocab_size > '
                             '--reinit_cardinality_threshold and rebuild the Adagrad '
                             'optimizer state (cold-restart trick for high-cardinality '
                             'features to reduce overfitting)')
    parser.add_argument('--reinit_cardinality_threshold', type=int, default=0,
                        help='Cardinality threshold used by the re-init strategy: '
                             'Embeddings whose vocab_size exceeds this value are reset '
                             'at each epoch end (0 = never reset any Embedding)')

    # Embedding construction control.
    parser.add_argument('--emb_skip_threshold', type=int, default=0,
                        help='At model construction time, features whose vocab_size '
                             'exceeds this value get no Embedding and are represented '
                             'by a zero vector at forward time (0 = no skipping; '
                             'all features get an Embedding). Useful for saving GPU '
                             'memory on ultra-high-cardinality features.')
    parser.add_argument('--seq_id_threshold', type=int, default=10000,
                        help='Within the sequence tokenizer, features with vocab_size '
                             'exceeding this value are treated as id features and receive '
                             'extra dropout(rate*2) during training to reduce overfitting. '
                             'Features at or below this threshold are treated as side-info '
                             'and receive no extra dropout.')

    _default_ns_groups = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'ns_groups.json')
    parser.add_argument('--ns_groups_json', type=str, default=_default_ns_groups,
                        help='Path to the NS-groups JSON file. If it does not exist, '
                             'each feature is placed in its own singleton group.')

    # NS tokenizer variant.
    parser.add_argument('--ns_tokenizer_type', type=str, default='rankmixer',
                        choices=['group', 'rankmixer'],
                        help='NS tokenizer variant: '
                             'group = project each group to one token, '
                             'rankmixer = concatenate all embeddings then split into '
                             'equal-size chunks (token count is tunable)')
    parser.add_argument('--user_ns_tokens', type=int, default=0,
                        help='Number of user NS tokens in rankmixer mode '
                             '(0 = automatically use the number of user groups)')
    parser.add_argument('--item_ns_tokens', type=int, default=0,
                        help='Number of item NS tokens in rankmixer mode '
                             '(0 = automatically use the number of item groups)')

    args = parser.parse_args()

    # Environment variables take precedence.
    args.data_dir = os.environ.get('TRAIN_DATA_PATH', args.data_dir)
    args.ckpt_dir = os.environ.get('TRAIN_CKPT_PATH', args.ckpt_dir)
    args.log_dir = os.environ.get('TRAIN_LOG_PATH', args.log_dir)
    args.tf_events_dir = os.environ.get('TRAIN_TF_EVENTS_PATH')

    return args


def main() -> None:
    args = parse_args()

    # Create output directories.
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)

    # Initialize logger and RNG.
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, 'train.log'))
    logging.info(f"Args: {vars(args)}")

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.tf_events_dir)

    # ---- Data loading ----
    if args.schema_path:
        schema_path = args.schema_path
    else:
        schema_path = os.path.join(args.data_dir, 'schema.json')

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"schema file not found at {schema_path}")
    _log_data_profile(args.data_dir, schema_path)

    # Parse per-domain sequence-length overrides.
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(','):
            k, v = pair.split(':')
            seq_max_lens[k.strip()] = int(v.strip())
        logging.info(f"Seq max_lens override: {seq_max_lens}")

    logging.info("Using Parquet data format (IterableDataset)")
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        seed=args.seed,
        seq_max_lens=seq_max_lens,
    )

    # ---- NS groups ----
    if args.ns_groups_json and os.path.exists(args.ns_groups_json):
        logging.info(f"Loading NS groups from {args.ns_groups_json}")
        with open(args.ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.user_int_schema.entries)}
        item_fid_to_idx = {fid: i for i, (fid, _, _) in enumerate(pcvr_dataset.item_int_schema.entries)}
        user_ns_groups = [[user_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['user_ns_groups'].values()]
        item_ns_groups = [[item_fid_to_idx[f] for f in fids] for fids in ns_groups_cfg['item_ns_groups'].values()]
        logging.info(f"User NS groups ({len(user_ns_groups)}): {list(ns_groups_cfg['user_ns_groups'].keys())}")
        logging.info(f"Item NS groups ({len(item_ns_groups)}): {list(ns_groups_cfg['item_ns_groups'].keys())}")
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(pcvr_dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(pcvr_dataset.item_int_schema.entries))]

    # ---- Build model ----
    user_int_feature_specs = build_feature_specs(
        pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes)

    model_args = {
        "user_int_feature_specs": user_int_feature_specs,
        "item_int_feature_specs": item_int_feature_specs,
        "user_dense_dim": pcvr_dataset.user_dense_schema.total_dim,
        "item_dense_dim": pcvr_dataset.item_dense_schema.total_dim,
        "seq_vocab_sizes": pcvr_dataset.seq_domain_vocab_sizes,
        "user_ns_groups": user_ns_groups,
        "item_ns_groups": item_ns_groups,
        "d_model": args.d_model,
        "emb_dim": args.emb_dim,
        "num_queries": args.num_queries,
        "num_hyformer_blocks": args.num_hyformer_blocks,
        "num_heads": args.num_heads,
        "seq_encoder_type": args.seq_encoder_type,
        "hidden_mult": args.hidden_mult,
        "dropout_rate": args.dropout_rate,
        "seq_top_k": args.seq_top_k,
        "seq_causal": args.seq_causal,
        "action_num": args.action_num,
        "num_time_buckets": NUM_TIME_BUCKETS if args.use_time_buckets else 0,
        "rank_mixer_mode": args.rank_mixer_mode,
        "use_rope": args.use_rope,
        "rope_base": args.rope_base,
        "emb_skip_threshold": args.emb_skip_threshold,
        "seq_id_threshold": args.seq_id_threshold,
        "ns_tokenizer_type": args.ns_tokenizer_type,
        "user_ns_tokens": args.user_ns_tokens,
        "item_ns_tokens": args.item_ns_tokens,
    }

    model = PCVRHyFormer(**model_args).to(args.device)

    # Log model sizing info.
    num_sequences = len(pcvr_dataset.seq_domains)
    num_ns = model.num_ns
    T = args.num_queries * num_sequences + num_ns
    logging.info(f"PCVRHyFormer model created: num_ns={num_ns}, T={T}, d_model={args.d_model}, rank_mixer_mode={args.rank_mixer_mode}")
    logging.info(f"User NS groups: {user_ns_groups}")
    logging.info(f"Item NS groups: {item_ns_groups}")
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")

    # ---- Training ----
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label='model',
    )

    ckpt_params = {
        "layer": args.num_hyformer_blocks,
        "head": args.num_heads,
        "hidden": args.d_model,
    }

    trainer = PCVRHyFormerRankingTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        num_epochs=args.num_epochs,
        device=args.device,
        save_dir=args.ckpt_dir,
        early_stopping=early_stopping,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        sparse_lr=args.sparse_lr,
        sparse_weight_decay=args.sparse_weight_decay,
        reinit_sparse_after_epoch=args.reinit_sparse_after_epoch,
        reinit_cardinality_threshold=args.reinit_cardinality_threshold,
        ckpt_params=ckpt_params,
        writer=writer,
        schema_path=schema_path,
        ns_groups_path=args.ns_groups_json if args.ns_groups_json and os.path.exists(args.ns_groups_json) else None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
    )

    trainer.train()
    writer.close()

    logging.info("Training complete!")


if __name__ == "__main__":
    main()
