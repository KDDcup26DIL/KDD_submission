import logging
from typing import Dict, List, NamedTuple, Set, Tuple

import torch
import torch.nn as nn


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict
    seq_lens: dict


class CrossNetV2(nn.Module):
    def __init__(self, input_dim: int, num_layers: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_layers)])

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x = x0
        for layer in self.kernels:
            x = x + x0 * layer(x)
        return x


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_units: List[int], dropout_rate: float) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = prev_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PooledEmbeddingBlock(nn.Module):
    def __init__(self, feature_specs: List[Tuple[int, int, int]], emb_dim: int, emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        raw_embs = []
        self.index_map: List[int] = []
        real_idx = 0
        for vocab_size, _, _ in feature_specs:
            skip = int(vocab_size) <= 0 or (emb_skip_threshold > 0 and int(vocab_size) > emb_skip_threshold)
            if skip:
                raw_embs.append(None)
                self.index_map.append(-1)
            else:
                raw_embs.append(nn.Embedding(int(vocab_size) + 1, emb_dim, padding_idx=0))
                self.index_map.append(real_idx)
                real_idx += 1
        self.embs = nn.ModuleList([emb for emb in raw_embs if emb is not None])
        self.emb_dim = emb_dim
        self.num_features = len(feature_specs)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pooled = []
        batch_size = feats.size(0)
        device = feats.device
        for feature_idx, (_, offset, length) in enumerate(self.feature_specs):
            emb_idx = self.index_map[feature_idx]
            if emb_idx == -1:
                pooled.append(torch.zeros(batch_size, self.emb_dim, device=device))
                continue
            emb = self.embs[emb_idx]
            if length == 1:
                vals = feats[:, offset].long().clamp(min=0)
                pooled.append(emb(vals))
            else:
                vals = feats[:, offset:offset + length].long().clamp(min=0)
                emb_all = emb(vals)
                mask = (vals != 0).float().unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled.append((emb_all * mask).sum(dim=1) / denom)
        return torch.cat(pooled, dim=-1)

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [param for emb in self.embs for param in emb.parameters()]

    def reinit_high_cardinality(self, threshold: int) -> Set[int]:
        reinit_ptrs: Set[int] = set()
        for feature_idx, (vocab_size, _, _) in enumerate(self.feature_specs):
            emb_idx = self.index_map[feature_idx]
            if emb_idx == -1:
                continue
            emb = self.embs[emb_idx]
            if emb.num_embeddings > threshold:
                nn.init.normal_(emb.weight, mean=0.0, std=0.01)
                with torch.no_grad():
                    emb.weight[0].zero_()
                reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs


class SequencePoolingBlock(nn.Module):
    def __init__(self, vocab_sizes: List[int], emb_dim: int, out_dim: int, emb_skip_threshold: int = 0) -> None:
        super().__init__()
        raw_embs = []
        self.index_map: List[int] = []
        real_idx = 0
        for vocab_size in vocab_sizes:
            skip = int(vocab_size) <= 0 or (emb_skip_threshold > 0 and int(vocab_size) > emb_skip_threshold)
            if skip:
                raw_embs.append(None)
                self.index_map.append(-1)
            else:
                raw_embs.append(nn.Embedding(int(vocab_size) + 1, emb_dim, padding_idx=0))
                self.index_map.append(real_idx)
                real_idx += 1
        self.embs = nn.ModuleList([emb for emb in raw_embs if emb is not None])
        self.emb_dim = emb_dim
        self.num_fields = len(vocab_sizes)
        self.out_proj = nn.Sequential(
            nn.Linear(self.num_fields * emb_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, seq_tensor: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        # seq_tensor: [B, S, L]
        pooled = []
        batch_size = seq_tensor.size(0)
        device = seq_tensor.device
        for slot in range(self.num_fields):
            emb_idx = self.index_map[slot]
            vals = seq_tensor[:, slot, :].long().clamp(min=0)
            if emb_idx == -1:
                pooled.append(torch.zeros(batch_size, self.emb_dim, device=device))
                continue
            emb = self.embs[emb_idx]
            emb_all = emb(vals)
            mask = (vals != 0).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled.append((emb_all * mask).sum(dim=1) / denom)
        flat = torch.cat(pooled, dim=-1)
        return self.out_proj(flat)

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [param for emb in self.embs for param in emb.parameters()]

    def reinit_high_cardinality(self, threshold: int) -> Set[int]:
        reinit_ptrs: Set[int] = set()
        for emb in self.embs:
            if emb.num_embeddings > threshold:
                nn.init.normal_(emb.weight, mean=0.0, std=0.01)
                with torch.no_grad():
                    emb.weight[0].zero_()
                reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs


class PCVRDCNv2(nn.Module):
    """DCNv2-style baseline adapted to the KDD submission I/O contract."""

    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        d_model: int = 64,
        emb_dim: int = 64,
        num_dcnv2_layers: int = 2,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        action_num: int = 1,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.emb_skip_threshold = emb_skip_threshold
        self.user_int_block = PooledEmbeddingBlock(user_int_feature_specs, emb_dim, emb_skip_threshold)
        self.item_int_block = PooledEmbeddingBlock(item_int_feature_specs, emb_dim, emb_skip_threshold)
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_blocks = nn.ModuleDict({
            domain: SequencePoolingBlock(vocab_sizes, emb_dim, d_model, emb_skip_threshold)
            for domain, vocab_sizes in seq_vocab_sizes.items()
        })

        self.user_sparse_proj = nn.Sequential(
            nn.Linear(len(user_int_feature_specs) * emb_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )
        self.item_sparse_proj = nn.Sequential(
            nn.Linear(len(item_int_feature_specs) * emb_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )

        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.ReLU(),
                nn.LayerNorm(d_model),
            )

        num_blocks = 2 + len(self.seq_domains) + int(self.has_user_dense) + int(self.has_item_dense)
        input_dim = num_blocks * d_model
        cross_layers = max(2, num_dcnv2_layers)
        self.crossnet = CrossNetV2(input_dim, cross_layers)
        dnn_hidden = [max(64, d_model * hidden_mult), max(32, d_model * 2)]
        self.parallel_dnn = MLPBlock(input_dim, dnn_hidden, dropout_rate)
        self.output = nn.Linear(input_dim + self.parallel_dnn.output_dim, action_num)
        self.reset_parameters()
        logging.info(f"PCVRDCNv2 created: input_dim={input_dim}, cross_layers={cross_layers}, d_model={d_model}, emb_dim={emb_dim}")

    @staticmethod
    def _sanitize_dense(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=-1e6, max=1e6)
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def _build_flat_features(self, inputs: ModelInput) -> torch.Tensor:
        parts = [
            self.user_sparse_proj(self.user_int_block(inputs.user_int_feats)),
            self.item_sparse_proj(self.item_int_block(inputs.item_int_feats)),
        ]
        if self.has_user_dense:
            parts.append(self.user_dense_proj(self._sanitize_dense(inputs.user_dense_feats)))
        if self.has_item_dense:
            parts.append(self.item_dense_proj(self._sanitize_dense(inputs.item_dense_feats)))
        for domain in self.seq_domains:
            parts.append(self.seq_blocks[domain](inputs.seq_data[domain], inputs.seq_lens[domain]))
        return torch.cat(parts, dim=-1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        flat = self._build_flat_features(inputs)
        cross_out = self.crossnet(flat)
        dnn_out = self.parallel_dnn(flat)
        logits = self.output(torch.cat([cross_out, dnn_out], dim=-1))
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        flat = self._build_flat_features(inputs)
        cross_out = self.crossnet(flat)
        dnn_out = self.parallel_dnn(flat)
        logits = self.output(torch.cat([cross_out, dnn_out], dim=-1))
        return logits, flat

    def get_sparse_params(self) -> List[nn.Parameter]:
        params = []
        params.extend(self.user_int_block.sparse_parameters())
        params.extend(self.item_int_block.sparse_parameters())
        for domain in self.seq_domains:
            params.extend(self.seq_blocks[domain].sparse_parameters())
        return params

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ids = {id(param) for param in self.get_sparse_params()}
        return [param for param in self.parameters() if id(param) not in sparse_ids]

    def reinit_high_cardinality_params(self, threshold: int) -> Set[int]:
        reinit_ptrs: Set[int] = set()
        reinit_ptrs.update(self.user_int_block.reinit_high_cardinality(threshold))
        reinit_ptrs.update(self.item_int_block.reinit_high_cardinality(threshold))
        for domain in self.seq_domains:
            reinit_ptrs.update(self.seq_blocks[domain].reinit_high_cardinality(threshold))
        logging.info(f"Re-initialized {len(reinit_ptrs)} high-cardinality embedding tensors (threshold={threshold})")
        return reinit_ptrs

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                with torch.no_grad():
                    module.weight[0].zero_()
