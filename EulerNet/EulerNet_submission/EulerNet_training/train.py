"""EulerNet training entry point for the KDD submission layout."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch

from dataset import FeatureSchema, NUM_TIME_BUCKETS, get_pcvr_data
from model import EulerNet
from trainer import EulerNetRankingTrainer
from utils import EarlyStopping, create_logger, set_seed


def build_feature_specs(schema: FeatureSchema, per_position_vocab_sizes: List[int]) -> List[Tuple[int, int, int]]:
    return [
        (max(per_position_vocab_sizes[offset:offset + length]), offset, length)
        for _, offset, length in schema.entries
    ]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EulerNet Training")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--schema_path", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--buffer_batches", type=int, default=20)
    parser.add_argument("--train_ratio", type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--eval_every_n_steps", type=int, default=0)
    parser.add_argument("--seq_max_lens", type=str, default="seq_a:256,seq_b:256,seq_c:512,seq_d:512")
    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--shape", type=parse_int_list, default=[52,52,52,52])
    parser.add_argument("--net_ex_dropout", type=float, default=0.1)
    parser.add_argument("--net_im_dropout", type=float, default=0.1)
    parser.add_argument("--layer_norm", action="store_true", default=True)
    parser.add_argument("--no_layer_norm", dest="layer_norm", action="store_false")
    parser.add_argument("--loss_type", type=str, default="bce", choices=["bce", "focal"])
    parser.add_argument("--focal_alpha", type=float, default=0.1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--sparse_lr", type=float, default=0.05)
    parser.add_argument("--sparse_weight_decay", type=float, default=0.0)
    parser.add_argument("--reinit_sparse_after_epoch", type=int, default=0)
    parser.add_argument("--reinit_cardinality_threshold", type=int, default=0)
    parser.add_argument("--ns_groups_json", type=str, default="")
    args = parser.parse_args()
    args.data_dir = os.environ.get("TRAIN_DATA_PATH", args.data_dir)
    args.ckpt_dir = os.environ.get("TRAIN_CKPT_PATH", args.ckpt_dir)
    args.log_dir = os.environ.get("TRAIN_LOG_PATH", args.log_dir)
    args.tf_events_dir = os.environ.get("TRAIN_TF_EVENTS_PATH")
    return args


def main() -> None:
    args = parse_args()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    if args.tf_events_dir:
        Path(args.tf_events_dir).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    create_logger(os.path.join(args.log_dir, "train.log"))
    logging.info(f"Args: {vars(args)}")
    schema_path = args.schema_path or os.path.join(args.data_dir, "schema.json")
    seq_max_lens = {}
    if args.seq_max_lens:
        for pair in args.seq_max_lens.split(","):
            key, value = pair.split(":")
            seq_max_lens[key.strip()] = int(value.strip())
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
    model = EulerNet(
        user_int_feature_specs=build_feature_specs(pcvr_dataset.user_int_schema, pcvr_dataset.user_int_vocab_sizes),
        item_int_feature_specs=build_feature_specs(pcvr_dataset.item_int_schema, pcvr_dataset.item_int_vocab_sizes),
        user_dense_dim=pcvr_dataset.user_dense_schema.total_dim,
        item_dense_dim=pcvr_dataset.item_dense_schema.total_dim,
        embedding_dim=args.embedding_dim,
        shape=args.shape,
        net_ex_dropout=args.net_ex_dropout,
        net_im_dropout=args.net_im_dropout,
        layer_norm=args.layer_norm,
    ).to(args.device)
    logging.info(f"EulerNet model created: fields={model.num_ns_attribute}, params={sum(p.numel() for p in model.parameters()):,}")
    early_stopping = EarlyStopping(
        checkpoint_path=os.path.join(args.ckpt_dir, "placeholder", "model.pt"),
        patience=args.patience,
        label="model",
    )
    trainer = EulerNetRankingTrainer(
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
        ckpt_params={"layer": len(args.shape), "head": 0, "hidden": args.embedding_dim},
        writer=None,
        schema_path=schema_path,
        ns_groups_path=None,
        eval_every_n_steps=args.eval_every_n_steps,
        train_config=vars(args),
    )
    trainer.train()
    logging.info("Training complete")


if __name__ == "__main__":
    main()
