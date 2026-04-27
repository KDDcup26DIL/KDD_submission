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


class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_units: List[int], output_dim: int, dropout_rate: float) -> None:
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
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

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


class FactorizationMachineBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int, embedding_dim: int, rank_k: int,
                 mlp_hidden_units: List[int], dropout_rate: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_features = output_features
        self.rank_k = rank_k
        self.input_features = input_features
        self.proj_y = nn.Parameter(torch.randn(self.input_features, self.rank_k))
        fm_out_dim = input_features * rank_k
        self.layer_norm = nn.LayerNorm(fm_out_dim)
        self.mlp = MLPBlock(
            input_dim=fm_out_dim,
            hidden_units=mlp_hidden_units,
            output_dim=output_features * embedding_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x.transpose(1, 2) @ self.proj_y
        fm_matrix = torch.bmm(x, projected)
        mlp_in = self.layer_norm(fm_matrix.flatten(start_dim=1))
        mlp_out = self.mlp(mlp_in)
        return mlp_out.view(-1, self.output_features, self.embedding_dim)


class LinearCompressionBlock(nn.Module):
    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.transpose(1, 2))
        return out.transpose(1, 2)


class WuKongLayer(nn.Module):
    def __init__(self, input_features: int, lcb_features: int, fmb_features: int, embedding_dim: int,
                 fmp_rank_k: int, fmb_mlp_units: List[int], dropout_rate: float, layer_norm: bool) -> None:
        super().__init__()
        self.fmb = FactorizationMachineBlock(
            input_features=input_features,
            output_features=fmb_features,
            embedding_dim=embedding_dim,
            rank_k=fmp_rank_k,
            mlp_hidden_units=fmb_mlp_units,
            dropout_rate=dropout_rate,
        )
        self.lcb = LinearCompressionBlock(input_features, lcb_features)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None
        out_features = lcb_features + fmb_features
        self.residual_proj = nn.Linear(input_features, out_features) if input_features != out_features else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fmb_out = self.fmb(x)
        lcb_out = self.lcb(x)
        out = torch.cat([fmb_out, lcb_out], dim=1)
        if self.residual_proj is not None:
            res = self.residual_proj(x.transpose(1, 2)).transpose(1, 2)
        else:
            res = x
        out = out + res
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


class PCVRWuKong(nn.Module):
    """WuKong-style baseline adapted to the KDD submission I/O contract."""

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
        num_wukong_layers: int = 3,
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
        lcb_features: int = 8,
        fmb_features: int = 8,
        fmb_mlp_units: List[int] = None,
        fmp_rank_k: int = 8,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if fmb_mlp_units is None:
            fmb_mlp_units = [32, 32]
        self.emb_skip_threshold = emb_skip_threshold
        self.user_int_block = PooledEmbeddingBlock(user_int_feature_specs, emb_dim, emb_skip_threshold)
        self.item_int_block = PooledEmbeddingBlock(item_int_feature_specs, emb_dim, emb_skip_threshold)
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.num_ns = int(user_ns_tokens) + int(item_ns_tokens) + (1 if user_dense_dim > 0 else 0)
        self.seq_blocks = nn.ModuleDict({
            domain: SequencePoolingBlock(vocab_sizes, emb_dim, emb_dim, emb_skip_threshold)
            for domain, vocab_sizes in seq_vocab_sizes.items()
        })

        self.has_user_dense = user_dense_dim > 0
        self.has_item_dense = item_dense_dim > 0
        if self.has_user_dense:
            self.user_dense_proj = nn.Sequential(
                nn.Linear(user_dense_dim, emb_dim),
                nn.ReLU(),
                nn.LayerNorm(emb_dim),
            )
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, emb_dim),
                nn.ReLU(),
                nn.LayerNorm(emb_dim),
            )

        num_tokens = 2 + len(self.seq_domains) + int(self.has_user_dense) + int(self.has_item_dense)
        output_features = lcb_features + fmb_features
        self.wukong_stack = nn.Sequential(*[
            WuKongLayer(
                input_features=num_tokens if i == 0 else output_features,
                lcb_features=lcb_features,
                fmb_features=fmb_features,
                embedding_dim=emb_dim,
                fmp_rank_k=fmp_rank_k,
                fmb_mlp_units=fmb_mlp_units,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
            )
            for i in range(num_wukong_layers)
        ])
        self.fc = MLPBlock(
            input_dim=output_features * emb_dim,
            hidden_units=[max(64, d_model * hidden_mult), max(32, d_model * 2)],
            output_dim=action_num,
            dropout_rate=dropout_rate,
        )
        self.reset_parameters()
        logging.info(
            f"PCVRWuKong created: tokens={num_tokens}, layers={num_wukong_layers}, emb_dim={emb_dim}, "
            f"lcb={lcb_features}, fmb={fmb_features}"
        )

    @staticmethod
    def _sanitize_dense(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=-1e6, max=1e6)
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def _build_tokens(self, inputs: ModelInput) -> torch.Tensor:
        parts = [
            self.user_int_block(inputs.user_int_feats).view(inputs.user_int_feats.size(0), -1, self.user_int_block.emb_dim).mean(dim=1, keepdim=True),
            self.item_int_block(inputs.item_int_feats).view(inputs.item_int_feats.size(0), -1, self.item_int_block.emb_dim).mean(dim=1, keepdim=True),
        ]
        if self.has_user_dense:
            parts.append(self.user_dense_proj(self._sanitize_dense(inputs.user_dense_feats)).unsqueeze(1))
        if self.has_item_dense:
            parts.append(self.item_dense_proj(self._sanitize_dense(inputs.item_dense_feats)).unsqueeze(1))
        for domain in self.seq_domains:
            parts.append(self.seq_blocks[domain](inputs.seq_data[domain], inputs.seq_lens[domain]).unsqueeze(1))
        return torch.cat(parts, dim=1)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        tokens = self._build_tokens(inputs)
        out = self.wukong_stack(tokens)
        logits = self.fc(out.flatten(start_dim=1))
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self._build_tokens(inputs)
        out = self.wukong_stack(tokens)
        logits = self.fc(out.flatten(start_dim=1))
        return logits, out.flatten(start_dim=1)

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
