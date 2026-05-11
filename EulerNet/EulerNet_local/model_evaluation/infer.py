"""EulerNet inference script for the KDD evaluation container."""

import json
import logging
import os
from typing import Any, Dict, List, NamedTuple, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import FeatureSchema, PCVRParquetDataset
from model import EulerNet, ModelInput


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_FALLBACK_SEQ_MAX_LENS = "seq_a:256,seq_b:256,seq_c:512,seq_d:512"
_FALLBACK_BATCH_SIZE = 128
_FALLBACK_NUM_WORKERS = 16


def build_feature_specs(schema: FeatureSchema, per_position_vocab_sizes: List[int]) -> List[Tuple[int, int, int]]:
    return [
        (max(per_position_vocab_sizes[offset:offset + length]), offset, length)
        for _, offset, length in schema.entries
    ]


def parse_seq_max_lens(value: str) -> Dict[str, int]:
    parsed = {}
    for pair in value.split(","):
        key, val = pair.split(":")
        parsed[key.strip()] = int(val.strip())
    return parsed


def load_train_config(model_dir: str) -> Dict[str, Any]:
    path = os.path.join(model_dir, "train_config.json")
    if not os.path.exists(path):
        logging.warning("train_config.json not found; using fallback EulerNet defaults")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def build_model(dataset: PCVRParquetDataset, cfg: Dict[str, Any], device: str) -> EulerNet:
    model = EulerNet(
        user_int_feature_specs=build_feature_specs(dataset.user_int_schema, dataset.user_int_vocab_sizes),
        item_int_feature_specs=build_feature_specs(dataset.item_int_schema, dataset.item_int_vocab_sizes),
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        embedding_dim=int(cfg.get("embedding_dim", 10)),
        shape=cfg.get("shape", [52]),
        net_ex_dropout=float(cfg.get("net_ex_dropout", 0.1)),
        net_im_dropout=float(cfg.get("net_im_dropout", 0.1)),
        layer_norm=bool(cfg.get("layer_norm", True)),
    ).to(device)
    return model


def get_ckpt_path(model_dir: str) -> str:
    for item in os.listdir(model_dir):
        if item.endswith(".pt"):
            return os.path.join(model_dir, item)
    raise FileNotFoundError(f"No *.pt checkpoint found under {model_dir}")


def batch_to_model_input(batch: Dict[str, Any], device: str) -> ModelInput:
    device_batch = {}
    for key, value in batch.items():
        device_batch[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    seq_data, seq_lens, seq_time_buckets = {}, {}, {}
    for domain in device_batch["_seq_domains"]:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f"{domain}_len"]
        batch_size, _, seq_len = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f"{domain}_time_bucket",
            torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
        )
    return ModelInput(
        user_int_feats=device_batch["user_int_feats"],
        item_int_feats=device_batch["item_int_feats"],
        user_dense_feats=device_batch["user_dense_feats"],
        item_dense_feats=device_batch["item_dense_feats"],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
    )


def main() -> None:
    model_dir = os.environ["MODEL_OUTPUT_PATH"]
    data_dir = os.environ["EVAL_DATA_PATH"]
    result_dir = os.environ["EVAL_RESULT_PATH"]
    os.makedirs(result_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    schema_path = os.path.join(model_dir, "schema.json")
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, "schema.json")
    cfg = load_train_config(model_dir)
    dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=int(cfg.get("batch_size", _FALLBACK_BATCH_SIZE)),
        seq_max_lens=parse_seq_max_lens(cfg.get("seq_max_lens", _FALLBACK_SEQ_MAX_LENS)),
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )
    model = build_model(dataset, cfg, device)
    model.load_state_dict(torch.load(get_ckpt_path(model_dir), map_location=device), strict=True)
    model.eval()
    num_workers = int(cfg.get("num_workers", _FALLBACK_NUM_WORKERS))
    loader_kwargs = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    all_probs, all_user_ids = [], []
    with torch.no_grad():
        for batch in loader:
            logits, _ = model.predict(batch_to_model_input(batch, device))
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
            all_probs.extend(probs)
            all_user_ids.extend(batch.get("user_id", []))
    with open(os.path.join(result_dir, "predictions.json"), "w") as f:
        json.dump({"predictions": dict(zip(all_user_ids, all_probs))}, f)
    logging.info(f"Saved {len(all_probs)} predictions")


if __name__ == "__main__":
    main()
