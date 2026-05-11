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


class FeatureTokenEmbedding(nn.Module):
    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        emb_dim: int,
        emb_skip_threshold: int = 0,
    ) -> None:
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

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        tokens = []
        batch_size = feats.size(0)
        device = feats.device
        for feature_idx, (_, offset, length) in enumerate(self.feature_specs):
            emb_idx = self.index_map[feature_idx]
            if emb_idx == -1:
                tokens.append(torch.zeros(batch_size, self.emb_dim, device=device))
                continue
            emb = self.embs[emb_idx]
            if length == 1:
                vals = feats[:, offset].long().clamp(min=0)
                tokens.append(emb(vals))
            else:
                vals = feats[:, offset:offset + length].long().clamp(min=0)
                emb_all = emb(vals)
                mask = (vals != 0).float().unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                tokens.append((emb_all * mask).sum(dim=1) / denom)
        return torch.stack(tokens, dim=1)

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


class SequenceTokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_sizes: List[int],
        emb_dim: int,
        emb_skip_threshold: int = 0,
    ) -> None:
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

    def forward(self, seq_tensor: torch.Tensor) -> torch.Tensor:
        # seq_tensor: [B, S, L], returns one pooled token per sequence field.
        tokens = []
        batch_size = seq_tensor.size(0)
        device = seq_tensor.device
        for slot in range(self.num_fields):
            emb_idx = self.index_map[slot]
            vals = seq_tensor[:, slot, :].long().clamp(min=0)
            if emb_idx == -1:
                tokens.append(torch.zeros(batch_size, self.emb_dim, device=device))
                continue
            emb_all = self.embs[emb_idx](vals)
            mask = (vals != 0).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            tokens.append((emb_all * mask).sum(dim=1) / denom)
        return torch.stack(tokens, dim=1)

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


class FeatureTokenizer(nn.Module):
    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        emb_dim: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.user_int = FeatureTokenEmbedding(user_int_feature_specs, emb_dim, emb_skip_threshold)
        self.item_int = FeatureTokenEmbedding(item_int_feature_specs, emb_dim, emb_skip_threshold)
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_blocks = nn.ModuleDict({
            domain: SequenceTokenEmbedding(vocab_sizes, emb_dim, emb_skip_threshold)
            for domain, vocab_sizes in seq_vocab_sizes.items()
        })
        self.has_user_dense = user_dense_dim > 0
        self.has_item_dense = item_dense_dim > 0
        self.user_dense_proj = nn.Linear(user_dense_dim, emb_dim) if self.has_user_dense else None
        self.item_dense_proj = nn.Linear(item_dense_dim, emb_dim) if self.has_item_dense else None

    @staticmethod
    def _sanitize_dense(x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=-1e6, max=1e6)
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        parts = [
            self.user_int(inputs.user_int_feats),
            self.item_int(inputs.item_int_feats),
        ]
        if self.has_user_dense:
            parts.append(self.user_dense_proj(self._sanitize_dense(inputs.user_dense_feats)).unsqueeze(1))
        if self.has_item_dense:
            parts.append(self.item_dense_proj(self._sanitize_dense(inputs.item_dense_feats)).unsqueeze(1))
        for domain in self.seq_domains:
            parts.append(self.seq_blocks[domain](inputs.seq_data[domain]))
        return torch.cat(parts, dim=1)

    def sparse_parameters(self) -> List[nn.Parameter]:
        params = []
        params.extend(self.user_int.sparse_parameters())
        params.extend(self.item_int.sparse_parameters())
        for domain in self.seq_domains:
            params.extend(self.seq_blocks[domain].sparse_parameters())
        return params

    def reinit_high_cardinality(self, threshold: int) -> Set[int]:
        reinit_ptrs: Set[int] = set()
        reinit_ptrs.update(self.user_int.reinit_high_cardinality(threshold))
        reinit_ptrs.update(self.item_int.reinit_high_cardinality(threshold))
        for domain in self.seq_domains:
            reinit_ptrs.update(self.seq_blocks[domain].reinit_high_cardinality(threshold))
        return reinit_ptrs


class HyFormerTower(nn.Module):
    """Compact HyFormer-style tower: feature/sequence tokens + learned query + Transformer encoder."""

    def __init__(self, emb_dim: int, d_model: int, num_layers: int, num_heads: int, hidden_mult: int, dropout_rate: float) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(emb_dim, d_model), nn.LayerNorm(d_model))
        self.query_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * hidden_mult,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, field_tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(field_tokens)
        query = self.query_token.expand(x.size(0), -1, -1)
        x = torch.cat([query, x], dim=1)
        x = self.encoder(x)
        return self.output(x[:, 0])


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_units: List[int], dropout_rate: float, output_activation: str = None) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if output_activation == 'relu':
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizationMachineBlock(nn.Module):
    """FuxiCTR WuKong FMB adapted to the local self-contained trainer."""

    def __init__(
        self,
        input_features: int,
        output_features: int,
        embedding_dim: int,
        rank_k: int = 8,
        mlp_hidden_units: List[int] = None,
        mlp_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_hidden_units = mlp_hidden_units or [32, 32]
        self.output_features = output_features
        self.rank_k = rank_k
        self.input_features = input_features
        self.proj_Y = nn.Parameter(torch.randn(input_features, rank_k))
        fm_out_dim = input_features * rank_k
        self.layer_norm = nn.LayerNorm(fm_out_dim)
        self.mlp = SimpleMLP(
            input_dim=fm_out_dim,
            output_dim=output_features * embedding_dim,
            hidden_units=mlp_hidden_units,
            dropout_rate=mlp_dropout,
            output_activation='relu',
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x.transpose(1, 2) @ self.proj_Y
        fm_matrix = torch.bmm(x, projected)
        mlp_in = self.layer_norm(fm_matrix.flatten(start_dim=1))
        mlp_out = self.mlp(mlp_in)
        return mlp_out.view(-1, self.output_features, x.size(-1))


class LinearCompressionBlock(nn.Module):
    """FuxiCTR WuKong LCB."""

    def __init__(self, input_features: int, output_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.transpose(1, 2)).transpose(1, 2)


class WuKongLayer(nn.Module):
    def __init__(
        self,
        input_features: int,
        lcb_features: int,
        fmb_features: int,
        embedding_dim: int,
        fmp_rank_k: int = 8,
        fmb_mlp_units: List[int] = None,
        fmb_dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.fmb = FactorizationMachineBlock(
            input_features=input_features,
            output_features=fmb_features,
            embedding_dim=embedding_dim,
            rank_k=fmp_rank_k,
            mlp_hidden_units=fmb_mlp_units or [32, 32],
            mlp_dropout=fmb_dropout,
        )
        self.lcb = LinearCompressionBlock(input_features, lcb_features)
        output_features = lcb_features + fmb_features
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None
        self.residual_proj = nn.Linear(input_features, output_features) if input_features != output_features else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.fmb(x), self.lcb(x)], dim=1)
        if self.residual_proj is None:
            res = x
        else:
            res = self.residual_proj(x.transpose(1, 2)).transpose(1, 2)
        out = out + res
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return out


class WuKongTower(nn.Module):
    def __init__(
        self,
        input_features: int,
        emb_dim: int,
        d_model: int,
        num_layers: int,
        lcb_features: int,
        fmb_features: int,
        fmp_rank_k: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        output_features = lcb_features + fmb_features
        self.wukong_stack = nn.Sequential(*[
            WuKongLayer(
                input_features=input_features if i == 0 else output_features,
                lcb_features=lcb_features,
                fmb_features=fmb_features,
                embedding_dim=emb_dim,
                fmp_rank_k=fmp_rank_k,
                fmb_mlp_units=[max(32, d_model // 2), max(32, d_model // 2)],
                fmb_dropout=dropout_rate,
                layer_norm=True,
            )
            for i in range(num_layers)
        ])
        self.output = nn.Sequential(
            nn.Linear(output_features * emb_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, field_tokens: torch.Tensor) -> torch.Tensor:
        x = self.wukong_stack(field_tokens)
        return self.output(x.flatten(start_dim=1))


class PCVRHyFormerWuKong(nn.Module):
    """Two-tower model: HyFormer-style transformer tower + FuxiCTR-style WuKong tower with simple fusion."""

    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: Dict[str, List[int]],
        d_model: int = 64,
        emb_dim: int = 64,
        num_hyformer_layers: int = 2,
        num_heads: int = 4,
        num_wukong_layers: int = 3,
        lcb_features: int = 32,
        fmb_features: int = 32,
        fmp_rank_k: int = 8,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        fusion_hidden: int = 128,
        action_num: int = 1,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.tokenizer = FeatureTokenizer(
            user_int_feature_specs=user_int_feature_specs,
            item_int_feature_specs=item_int_feature_specs,
            user_dense_dim=user_dense_dim,
            item_dense_dim=item_dense_dim,
            seq_vocab_sizes=seq_vocab_sizes,
            emb_dim=emb_dim,
            emb_skip_threshold=emb_skip_threshold,
        )
        num_fields = (
            len(user_int_feature_specs)
            + len(item_int_feature_specs)
            + int(user_dense_dim > 0)
            + int(item_dense_dim > 0)
            + sum(len(v) for v in seq_vocab_sizes.values())
        )
        self.hyformer_tower = HyFormerTower(
            emb_dim=emb_dim,
            d_model=d_model,
            num_layers=num_hyformer_layers,
            num_heads=num_heads,
            hidden_mult=hidden_mult,
            dropout_rate=dropout_rate,
        )
        self.wukong_tower = WuKongTower(
            input_features=num_fields,
            emb_dim=emb_dim,
            d_model=d_model,
            num_layers=num_wukong_layers,
            lcb_features=lcb_features,
            fmb_features=fmb_features,
            fmp_rank_k=fmp_rank_k,
            dropout_rate=dropout_rate,
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 4, fusion_hidden),
            nn.ReLU(),
            nn.LayerNorm(fusion_hidden),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden, action_num),
        )
        self.reset_parameters()
        logging.info(
            "PCVRHyFormerWuKong created: fields=%d, d_model=%d, emb_dim=%d, "
            "hyformer_layers=%d, wukong_layers=%d, lcb=%d, fmb=%d",
            num_fields, d_model, emb_dim, num_hyformer_layers, num_wukong_layers,
            lcb_features, fmb_features,
        )

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        tokens = self.tokenizer(inputs)
        hy = self.hyformer_tower(tokens)
        wk = self.wukong_tower(tokens)
        fused = torch.cat([hy, wk, hy * wk, torch.abs(hy - wk)], dim=-1)
        return self.fusion(fused)

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(inputs)
        hy = self.hyformer_tower(tokens)
        wk = self.wukong_tower(tokens)
        fused = torch.cat([hy, wk, hy * wk, torch.abs(hy - wk)], dim=-1)
        return self.fusion(fused), fused

    def get_sparse_params(self) -> List[nn.Parameter]:
        return self.tokenizer.sparse_parameters()

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ids = {id(param) for param in self.get_sparse_params()}
        return [param for param in self.parameters() if id(param) not in sparse_ids]

    def reinit_high_cardinality_params(self, threshold: int) -> Set[int]:
        reinit_ptrs = self.tokenizer.reinit_high_cardinality(threshold)
        logging.info("Re-initialized %d high-cardinality embedding tensors (threshold=%d)", len(reinit_ptrs), threshold)
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
        nn.init.normal_(self.hyformer_tower.query_token, mean=0.0, std=0.02)


# Keep the existing dcnv2 scaffold import path working.
PCVRDCNv2 = PCVRHyFormerWuKong
