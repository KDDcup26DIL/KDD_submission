"""Standalone EulerNet implementation for the KDD submission interface."""

from typing import List, NamedTuple, Tuple

import torch
import torch.nn as nn


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict
    seq_lens: dict
    seq_time_buckets: dict


class CTRFeatureEmbedding(nn.Module):
    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.user_specs = user_int_feature_specs
        self.item_specs = item_int_feature_specs
        self.sparse_vocab_sizes = [max(2, int(vocab_size) + 1) for vocab_size, _, _ in user_int_feature_specs + item_int_feature_specs]
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim, sparse=True)
            for vocab_size in self.sparse_vocab_sizes
        ])
        self.user_dense_dim = int(user_dense_dim)
        self.item_dense_dim = int(item_dense_dim)
        dense_dim = self.user_dense_dim + self.item_dense_dim
        self.dense_weight = nn.Parameter(torch.empty(dense_dim, embedding_dim)) if dense_dim > 0 else None
        self.dense_bias = nn.Parameter(torch.zeros(dense_dim, embedding_dim)) if dense_dim > 0 else None
        self.num_fields = len(self.embeddings) + dense_dim
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.embeddings:
            nn.init.xavier_uniform_(emb.weight)
        if self.dense_weight is not None:
            nn.init.xavier_uniform_(self.dense_weight)

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        outputs = []
        sparse_columns = torch.cat([inputs.user_int_feats.long(), inputs.item_int_feats.long()], dim=1)
        for idx, emb in enumerate(self.embeddings):
            values = sparse_columns[:, idx].clamp_min(0).clamp_max(emb.num_embeddings - 1)
            outputs.append(emb(values))

        dense_parts = []
        if self.user_dense_dim > 0:
            dense_parts.append(inputs.user_dense_feats.float())
        if self.item_dense_dim > 0:
            dense_parts.append(inputs.item_dense_feats.float())
        if dense_parts:
            dense_values = torch.cat(dense_parts, dim=1)
            dense_emb = dense_values.unsqueeze(-1) * self.dense_weight.unsqueeze(0) + self.dense_bias.unsqueeze(0)
            outputs.extend(torch.unbind(dense_emb, dim=1))

        return torch.stack(outputs, dim=1)

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [emb.weight for emb in self.embeddings]

    def reinit_high_cardinality_params(self, threshold: int) -> List[int]:
        if threshold <= 0:
            return []
        ptrs = []
        for vocab_size, emb in zip(self.sparse_vocab_sizes, self.embeddings):
            if vocab_size > threshold:
                nn.init.xavier_uniform_(emb.weight)
                ptrs.append(emb.weight.data_ptr())
        return ptrs


class EulerInteractionLayer(nn.Module):
    def __init__(self, inshape, outshape, embedding_dim, apply_norm, net_ex_dropout, net_im_dropout):
        super().__init__()
        self.feature_dim = embedding_dim
        self.apply_norm = apply_norm
        if inshape == outshape:
            init_orders = torch.eye(inshape // self.feature_dim, outshape // self.feature_dim)
        else:
            init_orders = torch.softmax(
                torch.randn(inshape // self.feature_dim, outshape // self.feature_dim) / 0.01,
                dim=0,
            )
        self.inter_orders = nn.Parameter(init_orders)
        self.im = nn.Linear(inshape, outshape)
        nn.init.xavier_uniform_(self.im.weight)
        self.bias_lam = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01)
        self.bias_theta = nn.Parameter(torch.randn(1, self.feature_dim, outshape // self.feature_dim) * 0.01)
        self.drop_ex = nn.Dropout(p=net_ex_dropout)
        self.drop_im = nn.Dropout(p=net_im_dropout)
        self.norm_r = nn.LayerNorm([self.feature_dim])
        self.norm_p = nn.LayerNorm([self.feature_dim])

    def forward(self, complex_features):
        r, p = complex_features
        lam = r ** 2 + p ** 2 + 1e-8
        theta = torch.atan2(p, r)
        lam = lam.reshape(lam.shape[0], -1, self.feature_dim)
        theta = theta.reshape(theta.shape[0], -1, self.feature_dim)
        lam = 0.5 * torch.log(lam)
        lam, theta = self.drop_ex(lam), self.drop_ex(theta)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)
        lam = lam @ self.inter_orders + self.bias_lam
        theta = theta @ self.inter_orders + self.bias_theta
        lam = torch.exp(lam)
        lam, theta = torch.transpose(lam, -2, -1), torch.transpose(theta, -2, -1)

        r, p = r.reshape(r.shape[0], -1), p.reshape(p.shape[0], -1)
        r, p = self.drop_im(r), self.drop_im(p)
        r, p = torch.relu(self.im(r)), torch.relu(self.im(p))
        r = r.reshape(r.shape[0], -1, self.feature_dim)
        p = p.reshape(p.shape[0], -1, self.feature_dim)

        o_r = r + lam * torch.cos(theta)
        o_p = p + lam * torch.sin(theta)
        if self.apply_norm:
            o_r, o_p = self.norm_r(o_r), self.norm_p(o_p)
        return o_r, o_p


class EulerNet(nn.Module):
    def __init__(
        self,
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        embedding_dim: int = 10,
        shape: List[int] = None,
        net_ex_dropout: float = 0.0,
        net_im_dropout: float = 0.0,
        layer_norm: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        shape = shape or [52]
        self.embedding_layer = CTRFeatureEmbedding(
            user_int_feature_specs,
            item_int_feature_specs,
            user_dense_dim,
            item_dense_dim,
            embedding_dim,
        )
        field_num = self.embedding_layer.num_fields
        shape_list = [embedding_dim * field_num] + [num_neurons * embedding_dim for num_neurons in shape]
        layers = [
            EulerInteractionLayer(inshape, outshape, embedding_dim, layer_norm, net_ex_dropout, net_im_dropout)
            for inshape, outshape in zip(shape_list[:-1], shape_list[1:])
        ]
        self.Euler_interaction_layers = nn.Sequential(*layers)
        self.mu = nn.Parameter(torch.ones(1, field_num, 1))
        self.reg = nn.Linear(shape_list[-1], 1)
        nn.init.xavier_normal_(self.reg.weight)
        self.num_ns_attribute = field_num

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        feature_emb = self.embedding_layer(inputs)
        r = self.mu * torch.cos(feature_emb)
        p = self.mu * torch.sin(feature_emb)
        o_r, o_p = self.Euler_interaction_layers((r, p))
        o_r = o_r.reshape(o_r.shape[0], -1)
        o_p = o_p.reshape(o_p.shape[0], -1)
        return self.reg(o_r) + self.reg(o_p)

    def predict(self, inputs: ModelInput):
        logits = self.forward(inputs)
        return logits, logits

    def get_sparse_params(self) -> List[nn.Parameter]:
        return self.embedding_layer.sparse_parameters()

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.requires_grad and p.data_ptr() not in sparse_ptrs]

    def reinit_high_cardinality_params(self, threshold: int) -> List[int]:
        return self.embedding_layer.reinit_high_cardinality_params(threshold)
