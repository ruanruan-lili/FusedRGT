import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# ---------------------------
# 特征融合模块
# ---------------------------
class MultiModalFeatureFusion(nn.Module):
    def __init__(self, feat_dims, common_dim=128):
        super().__init__()
        self.bool_proj = nn.Linear(feat_dims['bool'], common_dim)
        self.cat_proj = nn.Linear(feat_dims['cat'], common_dim)
        self.num_proj = nn.Linear(feat_dims['num'], common_dim)
        self.desc_proj = nn.Linear(feat_dims['desc'], common_dim)
        self.tweet_proj = nn.Linear(feat_dims['tweet'], common_dim)
        self.tweet_aug_proj = nn.Linear(feat_dims['tweet_aug'], common_dim)
        self.net_proj = nn.Linear(feat_dims['net'], common_dim)

        self.weights = nn.Parameter(torch.ones(7))
        self.softmax = nn.Softmax(dim=0)
        self.norm = nn.LayerNorm(common_dim)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.bool_proj, self.cat_proj, self.num_proj, self.desc_proj,
                      self.tweet_proj, self.tweet_aug_proj, self.net_proj]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.ones_(self.weights)

    def forward(self, feats):
        bool_emb = F.leaky_relu(self.bool_proj(feats['bool']))
        cat_emb = F.leaky_relu(self.cat_proj(feats['cat']))
        num_emb = F.leaky_relu(self.num_proj(feats['num']))
        desc_emb = F.leaky_relu(self.desc_proj(feats['desc']))
        tweet_emb = F.leaky_relu(self.tweet_proj(feats['tweet']))
        tweet_aug_emb = F.leaky_relu(self.tweet_aug_proj(feats['tweet_aug']))
        net_emb = F.leaky_relu(self.net_proj(feats['net']))

        all_embs = torch.stack([
            bool_emb, cat_emb, num_emb, desc_emb,
            tweet_emb, tweet_aug_emb, net_emb
        ], dim=0)

        norm_weights = self.softmax(self.weights)
        weighted_embs = norm_weights.view(7, 1, 1) * all_embs
        fused_emb = torch.sum(weighted_embs, dim=0)
        fused_emb = self.norm(fused_emb)
        return fused_emb

# ---------------------------
# 关系感知边丢弃
# ---------------------------
class RelationAwareEdgeDropout(nn.Module):
    def __init__(self, num_relations, drop_probs):
        super().__init__()
        assert isinstance(drop_probs, (list, tuple)) and len(drop_probs) == num_relations + 1
        self.num_relations = num_relations
        self.drop_probs = drop_probs

    def forward(self, edge_index, edge_type):
        if edge_index is None or edge_index.numel() == 0:
            device = edge_index.device if edge_index is not None else 'cpu'
            return torch.zeros((2, 0), dtype=torch.long, device=device), \
                   torch.zeros((0,), dtype=torch.long, device=device)
        edge_type = torch.clamp(edge_type.long(), 0, self.num_relations)
        mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

        for rel in range(len(self.drop_probs)):
            rel_mask = (edge_type == rel)
            if rel_mask.sum().item() == 0:
                continue
            rel_indices = rel_mask.nonzero(as_tuple=True)[0]
            keep_prob = 1 - self.drop_probs[rel]
            keep_mask = torch.rand(rel_indices.size(0), device=edge_index.device) < keep_prob
            mask[rel_indices] = mask[rel_indices] & keep_mask

        if mask.sum().item() == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=edge_index.device), \
                   torch.zeros((0,), dtype=torch.long, device=edge_index.device)
        edge_index = edge_index[:, mask]
        edge_type = edge_type[mask]
        return edge_index, edge_type

# ---------------------------
# RGT层
# ---------------------------
class RGTLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, heads=8, dropout=0.5):
        super().__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout  # float
        self.num_relations = num_relations

        self.Q = nn.Linear(in_channels, out_channels * heads)
        self.K = nn.Linear(in_channels, out_channels * heads)
        self.V = nn.Linear(in_channels, out_channels * heads)

        self.rel_k = nn.Embedding(num_relations, out_channels * heads)
        self.rel_v = nn.Embedding(num_relations, out_channels * heads)

        self.out_lin = nn.Linear(out_channels * heads, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.gate = nn.Linear(out_channels * 2, out_channels)
        self.sigmoid = nn.Sigmoid()

        self.ffn = nn.Sequential(
            nn.Linear(out_channels, out_channels * 4),
            nn.ReLU(),
            nn.Linear(out_channels * 4, out_channels),
            nn.Dropout(dropout)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.Q, self.K, self.V, self.out_lin]:
            if hasattr(layer, 'weight'):
                nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.rel_k.weight)
        nn.init.xavier_uniform_(self.rel_v.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for layer in self.ffn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, edge_index, edge_type):
        residual = x
        Q = self.Q(x).view(-1, self.heads, self.out_channels)
        K = self.K(x).view(-1, self.heads, self.out_channels)
        V = self.V(x).view(-1, self.heads, self.out_channels)

        out = self.propagate(edge_index=edge_index, Q=Q, K=K, V=V, edge_type=edge_type)
        out = self.out_lin(out.view(-1, self.heads * self.out_channels))

        gate_input = torch.cat([out, residual], dim=-1)
        gate = self.sigmoid(self.gate(gate_input))
        out = gate * out + (1 - gate) * residual

        out = self.norm1(out)
        ffn_out = self.ffn(out)

        gate_input = torch.cat([ffn_out, out], dim=-1)
        gate = self.sigmoid(self.gate(gate_input))
        out = gate * ffn_out + (1 - gate) * out

        return self.norm2(out)

    def message(self, Q_i, K_j, V_j, edge_type, index, ptr, size_i):
        rel_k = self.rel_k(edge_type).view(-1, self.heads, self.out_channels)
        rel_v = self.rel_v(edge_type).view(-1, self.heads, self.out_channels)
        K_j = K_j + rel_k
        V_j = V_j + rel_v
        alpha = (Q_i * K_j).sum(dim=-1) / (self.out_channels ** 0.5)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return V_j * alpha.unsqueeze(-1)

# ---------------------------
# RGT模型
# ---------------------------
class RGT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_layers=2, heads=8, dropout=0.5, edge_drop_probs=None):
        super().__init__()
        self.input_embed = nn.Linear(in_channels, hidden_channels)
        self.edge_dropout = RelationAwareEdgeDropout(num_relations, edge_drop_probs or [0.3]*(num_relations+1))
        self.layers = nn.ModuleList([
            RGTLayer(hidden_channels, hidden_channels, num_relations+1, heads, dropout)
            for _ in range(num_layers)
        ])
        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(hidden_channels * num_layers, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        self.layer_outputs = []

    def forward_embed(self, x, edge_index, edge_type):
        x = F.relu(self.input_embed(x))
        x = F.dropout(x, p=self.layers[0].dropout, training=self.training)
        edge_index, edge_type = self.edge_dropout(edge_index, edge_type)

        self.layer_outputs = [x]
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
            self.layer_outputs.append(x)

        if len(self.layer_outputs) > 1:
            fused_features = torch.cat(self.layer_outputs[1:], dim=-1)
            x = self.multi_scale_fusion(fused_features)
        return x

    def forward(self, x, edge_index, edge_type):
        emb = self.forward_embed(x, edge_index, edge_type)
        return self.classifier(emb)

# ---------------------------
# 融合特征的完整模型
# ---------------------------
class FusedRGT(nn.Module):
    def __init__(self, feat_dims, hidden_channels, out_channels, num_relations,
                 num_layers=2, heads=8, dropout=0.5, edge_drop_probs=None):
        super().__init__()
        self.feature_fusion = MultiModalFeatureFusion(feat_dims, hidden_channels)
        self.rgt = RGT(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            edge_drop_probs=edge_drop_probs
        )

    def forward(self, feats, edge_index, edge_type):
        fused_x = self.feature_fusion(feats)
        return self.rgt(fused_x, edge_index, edge_type)

    def forward_embed(self, feats, edge_index, edge_type):
        fused_x = self.feature_fusion(feats)
        return self.rgt.forward_embed(fused_x, edge_index, edge_type)
