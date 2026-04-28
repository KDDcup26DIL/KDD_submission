import logging
from typing import Dict, List, NamedTuple, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict
    seq_lens: dict
    seq_time_buckets: dict


class PooledEmbeddingBlock(nn.Module):
    def __init__(self, feature_specs: List[Tuple[int, int, int]], emb_dim: int, emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.emb_dim = emb_dim
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

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        batch_size = feats.size(0)
        device = feats.device
        pooled = []
        for feature_idx, (_, offset, length) in enumerate(self.feature_specs):
            emb_idx = self.index_map[feature_idx]
            if emb_idx == -1:
                pooled.append(torch.zeros(batch_size, self.emb_dim, device=device, dtype=torch.float32))
                continue
            emb = self.embs[emb_idx]
            if length == 1:
                pooled.append(emb(feats[:, offset].long().clamp(min=0)))
            else:
                vals = feats[:, offset:offset + length].long().clamp(min=0)
                emb_all = emb(vals)
                mask = (vals != 0).float().unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled.append((emb_all * mask).sum(dim=1) / denom)
        return torch.cat(pooled, dim=-1) if pooled else torch.zeros(batch_size, 0, device=device)

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [param for emb in self.embs for param in emb.parameters()]

    def reinit_high_cardinality(self, threshold: int) -> Set[int]:
        reinit_ptrs: Set[int] = set()
        for feature_idx, (vocab_size, _, _) in enumerate(self.feature_specs):
            emb_idx = self.index_map[feature_idx]
            if emb_idx == -1:
                continue
            emb = self.embs[emb_idx]
            if int(vocab_size) > threshold:
                nn.init.normal_(emb.weight, mean=0.0, std=0.01)
                with torch.no_grad():
                    emb.weight[0].zero_()
                reinit_ptrs.add(emb.weight.data_ptr())
        return reinit_ptrs


class SequenceMeanBlock(nn.Module):
    def __init__(self, vocab_sizes: List[int], emb_dim: int, emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.emb_dim = emb_dim
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

    def forward(self, seq_tensor: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        batch_size, num_fields, _ = seq_tensor.shape
        device = seq_tensor.device
        field_vecs = []
        for slot in range(num_fields):
            emb_idx = self.index_map[slot]
            if emb_idx == -1:
                field_vecs.append(torch.zeros(batch_size, self.emb_dim, device=device, dtype=torch.float32))
                continue
            vals = seq_tensor[:, slot, :].long().clamp(min=0)
            emb_all = self.embs[emb_idx](vals)
            mask = (vals != 0).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            field_vecs.append((emb_all * mask).sum(dim=1) / denom)
        return torch.cat(field_vecs, dim=-1) if field_vecs else torch.zeros(batch_size, 0, device=device)

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [param for emb in self.embs for param in emb.parameters()]

    def reinit_high_cardinality(self, threshold: int) -> Set[int]:
        reinit_ptrs: Set[int] = set()
        return reinit_ptrs


class SequenceLSTMBlock(nn.Module):
    def __init__(
        self,
        vocab_sizes: List[int],
        emb_dim: int,
        d_model: int,
        num_layers: int,
        dropout_rate: float,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
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
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=d_model,
            num_layers=max(1, num_layers),
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(self, seq_tensor: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        batch_size, num_fields, seq_len = seq_tensor.shape
        device = seq_tensor.device
        timestep_vecs = []
        for slot in range(num_fields):
            emb_idx = self.index_map[slot]
            if emb_idx == -1:
                timestep_vecs.append(torch.zeros(batch_size, seq_len, self.emb_dim, device=device, dtype=torch.float32))
                continue
            vals = seq_tensor[:, slot, :].long().clamp(min=0)
            timestep_vecs.append(self.embs[emb_idx](vals))
        if timestep_vecs:
            x = torch.stack(timestep_vecs, dim=0).mean(dim=0)
        else:
            x = torch.zeros(batch_size, seq_len, self.emb_dim, device=device, dtype=torch.float32)
        outputs, _ = self.lstm(x)
        idx = seq_lens.long().clamp(min=1, max=seq_len) - 1
        gather_idx = idx.view(batch_size, 1, 1).expand(-1, 1, outputs.size(-1))
        return outputs.gather(1, gather_idx).squeeze(1)

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [param for emb in self.embs for param in emb.parameters()]

    def reinit_high_cardinality(self, threshold: int) -> Set[int]:
        reinit_ptrs: Set[int] = set()
        return reinit_ptrs


class PCVRLSTM(nn.Module):
    """Raw LSTM baseline with pooled static features and sequence LSTMs."""

    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_lstm_layers: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_ns = int(user_ns_tokens) + int(item_ns_tokens) + (1 if user_dense_dim > 0 else 0)
        self.user_int_block = PooledEmbeddingBlock(user_int_feature_specs, emb_dim, emb_skip_threshold)
        self.item_int_block = PooledEmbeddingBlock(item_int_feature_specs, emb_dim, emb_skip_threshold)
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_blocks = nn.ModuleDict({
            domain: SequenceLSTMBlock(vocab_sizes, emb_dim, d_model, num_lstm_layers, dropout_rate, emb_skip_threshold)
            for domain, vocab_sizes in seq_vocab_sizes.items()
        })

        static_dim = (len(user_int_feature_specs) + len(item_int_feature_specs)) * emb_dim
        static_dim += user_dense_dim + item_dense_dim
        self.static_proj = nn.Sequential(
            nn.Linear(static_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_rate),
        )

        hidden_dim = max(d_model, d_model * hidden_mult)
        input_dim = d_model * (1 + len(self.seq_domains))
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, action_num),
        )
        self.reset_parameters()
        logging.info(f"PCVRLSTM created: static_dim={static_dim}, seq_domains={len(self.seq_domains)}, layers={num_lstm_layers}")

    @staticmethod
    def _sanitize_dense(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=-1e6, max=1e6)
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def _features(self, inputs: ModelInput) -> torch.Tensor:
        static_parts = [
            self.user_int_block(inputs.user_int_feats),
            self.item_int_block(inputs.item_int_feats),
            self._sanitize_dense(inputs.user_dense_feats),
            self._sanitize_dense(inputs.item_dense_feats),
        ]
        parts = [self.static_proj(torch.cat(static_parts, dim=-1))]
        for domain in self.seq_domains:
            parts.append(self.seq_blocks[domain](inputs.seq_data[domain], inputs.seq_lens[domain]))
        return torch.cat(parts, dim=-1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return self.head(self._features(inputs))

    def predict(self, inputs: ModelInput):
        feats = self._features(inputs)
        return self.head(feats), feats

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

