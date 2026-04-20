from __future__ import annotations

import os
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    P_i_random_list,
    A_j_random_list,
    A_m_random_list,
    bw_ru_du_random_list,
    bw_du_cu_random_list,
    max_RBs_per_UE,
    P_ib_sk_val,
)

# ============================ Hyperparameters ==========================
MAX_EPISODE = 50000
DISCOUNT_FACTOR = 0.99
GAE_LAMBDA = 0.95

STEPS_PER_UPDATE = 4096
MINIBATCH_SIZE = 512
WEIGHT_DECAY = 1e-4
ADAM_EPS = 1e-5

CLIP_RATIO = 0.20
UPDATE_EPOCHS = 4
VALUE_COEF = 0.5

ENTROPY_COEF_START = 0.02
ENTROPY_COEF_END = 0.003
ENTROPY_ANNEAL_EP = int(0.8 * MAX_EPISODE)

TARGET_KL = 0.01
MAX_GRAD_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True if DEVICE == "cuda" else False
TORCH_BACKEND_BENCHMARK = True


def set_torch_speed_flags():
    torch.backends.cudnn.benchmark = TORCH_BACKEND_BENCHMARK
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


# ===================================================
@dataclass
class CachedState:
    node_feats: torch.Tensor
    edge_feats: torch.Tensor
    request_feats: torch.Tensor
    orig_ids: List[int]
    ue_meta: List[dict]
    ru_mask: torch.Tensor
    du_mask: torch.Tensor
    cu_mask: torch.Tensor
    ru_valid_mask: torch.Tensor
    du_valid_mask: torch.Tensor
    cu_valid_mask: torch.Tensor


@dataclass
class CachedAction:
    env_action: Tuple[int, int, int, int, int, int, float]
    req_idx: int
    handover_flag: int
    ru_idx: int
    du_idx: int
    cu_idx: int
    rb_idx: int
    power_idx: int


@dataclass
class RolloutBuffer:
    states: List[CachedState]
    actions: List[CachedAction]
    logprobs: List[float]
    rewards: List[float]
    values: List[float]
    next_values: List[float]
    masks: List[int]


# =========================================================
# Init helper
# =========================================================
def orth_init(m: nn.Module, gain: float):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =========================================================
# GraphStateBuilder
# =========================================================
class GraphStateBuilder:
    def __init__(self, node_in, edge_in, request_in, device):
        self.node_in = int(node_in)
        self.edge_in = int(edge_in)
        self.request_in = int(request_in)
        self.device = device

    def _to_np(self, x):
        return np.asarray(x, dtype=np.float32)

    def _infer_sizes(self, state):
        RAN = state.get("RAN", {})
        RU = self._to_np(RAN.get("RU_power_remaining", []))
        DU = self._to_np(RAN.get("DU_remaining", []))
        CU = self._to_np(RAN.get("CU_remaining", []))
        LRU = self._to_np(RAN.get("link_bw_ru_du_bps", []))
        LDU = self._to_np(RAN.get("link_bw_du_cu_bps", []))
        return len(RU), len(DU), len(CU), RU, DU, CU, LRU, LDU

    def _norm_index(self, idx, n):
        if idx is None or idx < 0 or n <= 1:
            return 0.0
        return float(idx) / float(n - 1)

    def build_node_masks(self, num_RUs, num_DUs, num_CUs):
        total_nodes = num_RUs + num_DUs + num_CUs
        ru_mask = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)
        du_mask = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)
        cu_mask = torch.zeros(total_nodes, dtype=torch.bool, device=self.device)

        ru_mask[:num_RUs] = True
        du_mask[num_RUs:num_RUs + num_DUs] = True
        cu_mask[num_RUs + num_DUs:] = True
        return ru_mask, du_mask, cu_mask

    def _node_type_onehot(self, num_RUs, num_DUs, num_CUs):
        total_nodes = num_RUs + num_DUs + num_CUs
        t = torch.zeros((total_nodes, 3), dtype=torch.float32, device=self.device)
        t[:num_RUs, 0] = 1.0
        t[num_RUs:num_RUs + num_DUs, 1] = 1.0
        t[num_RUs + num_DUs:, 2] = 1.0
        return t

    def build_node_features(self, state):
        num_RUs, num_DUs, num_CUs, RU, DU, CU, _, _ = self._infer_sizes(state)

        ru_scale = max(float(np.max(P_i_random_list)), 1.0)
        du_scale = max(float(np.max(A_j_random_list)), 1.0)
        cu_scale = max(float(np.max(A_m_random_list)), 1.0)

        RU_n = RU / (ru_scale + 1e-9) if RU.size else np.zeros((0,), dtype=np.float32)
        DU_n = DU / (du_scale + 1e-9) if DU.size else np.zeros((0,), dtype=np.float32)
        CU_n = CU / (cu_scale + 1e-9) if CU.size else np.zeros((0,), dtype=np.float32)

        node_scalar = np.concatenate([RU_n, DU_n, CU_n], axis=0).astype(np.float32)
        node_feats = torch.from_numpy(node_scalar).to(self.device)

        if node_feats.numel() > 1:
            node_feats = (node_feats - node_feats.mean()) / (node_feats.std() + 1e-6)
        else:
            node_feats = node_feats * 0.0

        node_feats = node_feats.view(num_RUs + num_DUs + num_CUs, 1)
        node_type = self._node_type_onehot(num_RUs, num_DUs, num_CUs)
        node_feats = torch.cat([node_feats, node_type], dim=-1)
        return node_feats

    def build_edge_features(self, state):
        num_RUs, num_DUs, num_CUs, _, _, _, LRU, LDU = self._infer_sizes(state)
        total_nodes = num_RUs + num_DUs + num_CUs

        adj = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        if total_nodes > 0:
            np.fill_diagonal(adj, 1.0)

        ru_du_max = max(float(np.max(bw_ru_du_random_list)), 1.0)
        du_cu_max = max(float(np.max(bw_du_cu_random_list)), 1.0)

        if LRU.size > 0:
            nRU = min(num_RUs, LRU.shape[0])
            nDU = min(num_DUs, LRU.shape[1])
            for i in range(nRU):
                for j in range(nDU):
                    bw = float(LRU[i, j]) / (ru_du_max + 1e-9)
                    adj[i, num_RUs + j] = bw
                    adj[num_RUs + j, i] = bw

        if LDU.size > 0:
            nDU = min(num_DUs, LDU.shape[0])
            nCU = min(num_CUs, LDU.shape[1])
            for j in range(nDU):
                for k in range(nCU):
                    bw = float(LDU[j, k]) / (du_cu_max + 1e-9)
                    adj[num_RUs + j, num_RUs + num_DUs + k] = bw
                    adj[num_RUs + num_DUs + k, num_RUs + j] = bw

        return torch.from_numpy(adj).to(self.device).unsqueeze(-1)

    def build_request_features(self, state):
        UEs = list(state.get("UE_requests", {}).values())
        num_RUs, num_DUs, num_CUs, _, _, _, _, _ = self._infer_sizes(state)

        request_feats = []
        orig_ids = []
        ue_meta = []

        for idx, UE in enumerate(UEs):
            active = int(UE.get("status", {}).get("active", UE.get("active", 0)))
            if active != 1:
                continue

            prev_alloc = UE.get("prev_allocation", {})
            has_prev = int(prev_alloc.get("RU") is not None)

            prev_RU = int(prev_alloc["RU"]) if prev_alloc.get("RU") is not None else -1
            prev_DU = int(prev_alloc["DU"]) if prev_alloc.get("DU") is not None else -1
            prev_CU = int(prev_alloc["CU"]) if prev_alloc.get("CU") is not None else -1

            gains = np.asarray(UE.get("gain", []), dtype=np.float32)
            if gains.size > 0:
                top1 = float(np.max(gains))
                top2 = float(np.partition(gains, -2)[-2]) if gains.size >= 2 else 0.0
                gap = max(top1 - top2, 0.0)
                prev_gain = float(gains[prev_RU]) if 0 <= prev_RU < gains.size else 0.0
                best_RU = int(np.argmax(gains))
            else:
                top1 = top2 = gap = prev_gain = 0.0
                best_RU = -1

            R_min = float(UE.get("R_min", 0.0))
            delay = float(UE.get("delay", 1e-3))
            eta_slice = float(UE.get("eta_slice", 0.0))

            w_acc = float(UE.get("weight_accept", 1.0))
            w_thr = float(UE.get("weight_throughput", 1.0))
            w_lat = float(UE.get("weight_latency", 1.0))
            w_ho = float(UE.get("weight_handover", 1.0))

            pingpong = float(UE.get("pingpong", 0))
            handover_count = float(UE.get("handover_count", 0))

            feat = [
                R_min / 1e8,
                delay / 5e-3,
                w_acc,
                w_thr,
                w_lat,
                w_ho,
                eta_slice,
                np.log1p(max(top1, 0.0)),
                np.log1p(max(gap, 0.0)),
                np.log1p(max(prev_gain, 0.0)),
                float(has_prev),
                min(pingpong, 20.0) / 20.0,
                min(handover_count, 20.0) / 20.0,
                self._norm_index(prev_RU, num_RUs),
                self._norm_index(prev_DU, num_DUs),
                self._norm_index(prev_CU, num_CUs),
            ]

            request_feats.append(feat)
            orig_ids.append(int(UE.get("id", idx)))
            ue_meta.append({
                "id": int(UE.get("id", idx)),
                "has_prev": bool(has_prev),
                "prev_RU": prev_RU,
                "prev_DU": prev_DU,
                "prev_CU": prev_CU,
            })

        if len(request_feats) > 0:
            request_feats = torch.tensor(np.asarray(request_feats, dtype=np.float32), device=self.device)
        else:
            request_feats = torch.zeros((0, self.request_in), dtype=torch.float32, device=self.device)
            orig_ids = []
            ue_meta = []

        return request_feats, orig_ids, ue_meta

    def build_graph_state(self, state):
        num_RUs, num_DUs, num_CUs, _, _, _, _, _ = self._infer_sizes(state)
        node_feats = self.build_node_features(state)
        edge_feats = self.build_edge_features(state)
        req_feats, ids, ue_meta = self.build_request_features(state)
        ru_mask, du_mask, cu_mask = self.build_node_masks(num_RUs, num_DUs, num_CUs)

        RAN = state.get("RAN", {})
        ru_rem = np.asarray(RAN.get("RU_power_remaining", []), dtype=np.float32)
        du_rem = np.asarray(RAN.get("DU_remaining", []), dtype=np.float32)
        cu_rem = np.asarray(RAN.get("CU_remaining", []), dtype=np.float32)

        ru_valid_mask = torch.tensor(ru_rem > 1e-9, dtype=torch.bool, device=self.device)
        du_valid_mask = torch.tensor(du_rem > 1e-9, dtype=torch.bool, device=self.device)
        cu_valid_mask = torch.tensor(cu_rem > 1e-9, dtype=torch.bool, device=self.device)

        return CachedState(
            node_feats=node_feats,
            edge_feats=edge_feats,
            request_feats=req_feats,
            orig_ids=ids,
            ue_meta=ue_meta,
            ru_mask=ru_mask,
            du_mask=du_mask,
            cu_mask=cu_mask,
            ru_valid_mask=ru_valid_mask,
            du_valid_mask=du_valid_mask,
            cu_valid_mask=cu_valid_mask,
        )


# =========================================================
# Encoders
# =========================================================
class GraphSAGEEncoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.node_lin = nn.Linear(node_in, self.hidden_dim)
        self.edge_lin = nn.Linear(edge_in, self.hidden_dim)
        self.msg_lin = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.update_lin = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])
        self.act = nn.ReLU()

        self.apply(lambda m: orth_init(m, math.sqrt(2)))

    def forward(self, node_feats, edge_feats):
        h = self.act(self.node_lin(node_feats))
        e = self.act(self.edge_lin(edge_feats))

        N = h.shape[0]
        edge_weight = edge_feats.squeeze(-1).float().clamp(min=0.0)

        for layer in range(self.num_layers):
            sender = h.unsqueeze(1).expand(-1, N, -1)
            receiver = h.unsqueeze(0).expand(N, -1, -1)

            msg_input = torch.cat([sender, receiver, e], dim=-1)
            msgs = self.act(self.msg_lin(msg_input))
            msgs = msgs * edge_weight.unsqueeze(-1)

            deg = edge_weight.sum(dim=1, keepdim=True).clamp(min=1.0)
            agg = msgs.sum(dim=1) / deg

            h_new = self.update_lin(torch.cat([h, agg], dim=-1))
            h = h + self.act(h_new)
            h = self.norms[layer](h)

        graph_emb = h.mean(dim=0)
        return h, graph_emb


class MLPEncode(nn.Module):
    def __init__(self, req_in, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(req_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                orth_init(m, math.sqrt(2))

    def forward(self, request_feats):
        return self.net(request_feats)


# =========================================================
# Policy
# =========================================================
class FullPolicy(nn.Module):
    def __init__(self, node_in, edge_in, req_in, hidden_dim, num_graph_layers, max_RBs_per_UE, num_power_levels):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_RBs_per_UE = int(max_RBs_per_UE)
        self.num_power_levels = int(num_power_levels)

        self.graph_encoder = GraphSAGEEncoder(node_in, edge_in, hidden_dim, num_graph_layers)
        self.req_encoder = MLPEncode(req_in, hidden_dim)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        for m in self.fusion:
            if isinstance(m, nn.Linear):
                orth_init(m, math.sqrt(2))

        self.req_head = nn.Linear(hidden_dim, 1)
        self.handover_head = nn.Linear(hidden_dim, 2)

        self.node_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        self.rb_head = nn.Linear(hidden_dim, max_RBs_per_UE)
        self.power_head = nn.Linear(hidden_dim, num_power_levels)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        for m in [self.req_head, self.handover_head, self.node_proj, self.query_proj, self.rb_head, self.power_head]:
            orth_init(m, 0.01)
        for m in self.value_head:
            if isinstance(m, nn.Linear):
                orth_init(m, math.sqrt(2))

    def forward(self, graph_state):
        node_feats = graph_state.node_feats
        edge_feats = graph_state.edge_feats
        request_feats = graph_state.request_feats

        node_emb, graph_emb = self.graph_encoder(node_feats, edge_feats)

        device = node_feats.device
        M = request_feats.shape[0]

        if M == 0:
            empty_req = torch.zeros((0,), dtype=torch.float32, device=device)
            empty_req_emb = torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=device)
            empty_fused = torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=device)
            value_in = torch.cat([graph_emb, graph_emb], dim=-1)
            value = self.value_head(value_in).squeeze(-1)
            return node_emb, graph_emb, empty_req_emb, empty_fused, empty_req, value

        req_emb = self.req_encoder(request_feats)
        pooled_req = req_emb.mean(dim=0, keepdim=True).expand(M, -1)
        graph_ctx = graph_emb.unsqueeze(0).expand(M, -1)

        fused = self.fusion(torch.cat([req_emb, graph_ctx, pooled_req], dim=-1))
        req_logits = self.req_head(fused).squeeze(-1)

        value_in = torch.cat([graph_emb, req_emb.mean(dim=0)], dim=-1)
        value = self.value_head(value_in).squeeze(-1)

        return node_emb, graph_emb, req_emb, fused, req_logits, value

    def score_handover(self, req_vec):
        return self.handover_head(req_vec)

    def score_nodes(self, node_emb, req_vec, mask):
        idx = torch.where(mask)[0]
        if idx.numel() == 0:
            return idx, torch.zeros((0,), dtype=torch.float32, device=node_emb.device)

        q = self.query_proj(req_vec)
        k = self.node_proj(node_emb[idx])
        logits = (k * q.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.hidden_dim)
        return idx, logits

    def score_rb(self, req_vec):
        return self.rb_head(req_vec)

    def score_power(self, req_vec):
        return self.power_head(req_vec)


# =========================================================
# PPO Agent
# =========================================================
class PPOAgent:
    def __init__(
        self,
        learning_rate,
        device=DEVICE,
        power_levels=P_ib_sk_val,
        max_RBs_per_UE=max_RBs_per_UE,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        gamma=DISCOUNT_FACTOR,
        gae_lambda=GAE_LAMBDA,
        clip_ratio=CLIP_RATIO,
        value_coef=VALUE_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
    ):
        set_torch_speed_flags()

        self.device = torch.device(device)
        self.power_levels = list(power_levels) if len(power_levels) > 0 else [0.0]
        self.num_power_levels = len(self.power_levels)

        self.node_in = 4
        self.edge_in = 1
        self.request_in = 16
        self.hidden_dim = 128
        self.num_graph_layers = 2

        self.policy = FullPolicy(
            node_in=self.node_in,
            edge_in=self.edge_in,
            req_in=self.request_in,
            hidden_dim=self.hidden_dim,
            num_graph_layers=self.num_graph_layers,
            max_RBs_per_UE=max_RBs_per_UE,
            num_power_levels=self.num_power_levels,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            eps=ADAM_EPS,
            weight_decay=WEIGHT_DECAY,
        )

        self.scaler = torch.amp.GradScaler(enabled=USE_AMP)

        self.graph_builder = GraphStateBuilder(
            self.node_in,
            self.edge_in,
            self.request_in,
            self.device,
        )

        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.target_kl = float(target_kl)

        self.start_entropy = float(ENTROPY_COEF_START)
        self.end_entropy = float(ENTROPY_COEF_END)
        self.entropy_anneal_episodes = int(ENTROPY_ANNEAL_EP)
        self.entropy_coef = self.start_entropy

    def update_exploration(self, episode_idx):
        if self.entropy_anneal_episodes <= 0:
            self.entropy_coef = self.end_entropy
            return
        frac = min(1.0, episode_idx / float(self.entropy_anneal_episodes))
        self.entropy_coef = float(self.start_entropy + (self.end_entropy - self.start_entropy) * frac)

    def _sample_from_group(self, logits_1d, group_name):
        if logits_1d.numel() == 0:
            raise ValueError(f"No valid nodes in group {group_name}")
        dist = Categorical(logits=logits_1d)
        idx_t = dist.sample()
        return int(idx_t.item()), dist.log_prob(idx_t), dist.entropy()

    def select_action(self, state):
        graph_state = self.graph_builder.build_graph_state(state)

        if graph_state.request_feats.shape[0] == 0:
            return graph_state, None, None, 0.0

        self.policy.eval()
        with torch.no_grad():
            node_emb, graph_emb, req_emb, fused, req_logits, value = self.policy(graph_state)

            req_dist = Categorical(logits=req_logits)
            req_idx_t = req_dist.sample()
            req_idx = int(req_idx_t.item())
            ue_id = int(graph_state.orig_ids[req_idx])
            req_vec = fused[req_idx]
            meta = graph_state.ue_meta[req_idx]

            logprob = req_dist.log_prob(req_idx_t)

            has_prev = bool(meta["has_prev"])
            prev_RU = int(meta["prev_RU"])
            prev_DU = int(meta["prev_DU"])
            prev_CU = int(meta["prev_CU"])

            ru_valid = graph_state.ru_mask & graph_state.ru_valid_mask
            du_valid = graph_state.du_mask & graph_state.du_valid_mask
            cu_valid = graph_state.cu_mask & graph_state.cu_valid_mask

            alt_ru_available = False
            if has_prev and 0 <= prev_RU < ru_valid.numel():
                tmp = ru_valid.clone()
                tmp[prev_RU] = False
                alt_ru_available = bool(tmp.any().item())

            if has_prev and alt_ru_available:
                ho_logits = self.policy.score_handover(req_vec)
                ho_dist = Categorical(logits=ho_logits)
                ho_t = ho_dist.sample()
                handover_flag = int(ho_t.item())
                logprob = logprob + ho_dist.log_prob(ho_t)
            else:
                handover_flag = 0

            if has_prev and handover_flag == 0:
                RU_choice = prev_RU
                DU_choice = prev_DU
                CU_choice = prev_CU
            else:
                if has_prev and handover_flag == 1 and 0 <= prev_RU < ru_valid.numel():
                    ru_valid = ru_valid.clone()
                    ru_valid[prev_RU] = False

                ru_idx_pool, ru_logits = self.policy.score_nodes(node_emb, req_vec, ru_valid)
                du_idx_pool, du_logits = self.policy.score_nodes(node_emb, req_vec, du_valid)
                cu_idx_pool, cu_logits = self.policy.score_nodes(node_emb, req_vec, cu_valid)

                if ru_idx_pool.numel() == 0 or du_idx_pool.numel() == 0 or cu_idx_pool.numel() == 0:
                    return graph_state, None, None, 0.0

                ru_pos, lp_ru, _ = self._sample_from_group(ru_logits, "RU")
                du_pos, lp_du, _ = self._sample_from_group(du_logits, "DU")
                cu_pos, lp_cu, _ = self._sample_from_group(cu_logits, "CU")

                RU_choice = int(ru_idx_pool[ru_pos].item())
                DU_choice = int(du_idx_pool[du_pos].item())
                CU_choice = int(cu_idx_pool[cu_pos].item())

                logprob = logprob + lp_ru + lp_du + lp_cu

            rb_logits = self.policy.score_rb(req_vec)
            pw_logits = self.policy.score_power(req_vec)

            rb_idx, lp_rb, _ = self._sample_from_group(rb_logits, "RB")
            pw_idx, lp_pw, _ = self._sample_from_group(pw_logits, "Power")

            logprob = logprob + lp_rb + lp_pw

            num_RB_alloc = rb_idx + 1
            power_alloc = float(self.power_levels[pw_idx])

            env_action = (
                ue_id,
                int(handover_flag),
                int(RU_choice),
                int(DU_choice),
                int(CU_choice),
                int(num_RB_alloc),
                float(power_alloc),
            )

            cached_action = CachedAction(
                env_action=env_action,
                req_idx=req_idx,
                handover_flag=int(handover_flag),
                ru_idx=int(RU_choice),
                du_idx=int(DU_choice),
                cu_idx=int(CU_choice),
                rb_idx=int(rb_idx),
                power_idx=int(pw_idx),
            )

            return graph_state, cached_action, float(logprob.item()), float(value.item())

    def get_value(self, graph_state):
        self.policy.eval()
        with torch.no_grad():
            _, _, _, _, _, value = self.policy(graph_state)
        return float(value.item())

    def compute_GAE(self, rewards, values, next_values, masks):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_GAE = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * masks[t] - values[t]
            last_GAE = delta + self.gamma * self.gae_lambda * masks[t] * last_GAE
            advantages[t] = last_GAE

        returns = advantages + np.asarray(values, dtype=np.float32)
        return returns, advantages

    def update(self, buffer, episode_idx):
        if len(buffer.rewards) == 0:
            return

        self.update_exploration(episode_idx)
        self.policy.train()

        rewards = np.array(buffer.rewards, dtype=np.float32)
        values = np.array(buffer.values, dtype=np.float32)
        next_values = np.array(buffer.next_values, dtype=np.float32)
        masks = np.array(buffer.masks, dtype=np.float32)

        returns, advantages = self.compute_GAE(rewards, values, next_values, masks)

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        if adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        old_logp_t = torch.tensor(buffer.logprobs, dtype=torch.float32, device=self.device)
        old_value_t = torch.tensor(buffer.values, dtype=torch.float32, device=self.device)

        idxs = np.arange(len(buffer.rewards))

        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            approx_kls = []

            for start in range(0, len(idxs), self.minibatch_size):
                batch_idx = idxs[start:start + self.minibatch_size]
                if len(batch_idx) == 0:
                    continue

                policy_losses = []
                value_losses = []
                entropies = []
                kls = []

                self.optimizer.zero_grad(set_to_none=True)

                for bi in batch_idx:
                    state = buffer.states[bi]
                    action = buffer.actions[bi]
                    old_lp = old_logp_t[bi]
                    ret = ret_t[bi]
                    adv = adv_t[bi]
                    v_old = old_value_t[bi]

                    if state.request_feats.shape[0] == 0:
                        continue

                    node_emb, graph_emb, req_emb, fused, req_logits, v_pred = self.policy(state)
                    req_vec = fused[action.req_idx]
                    meta = state.ue_meta[action.req_idx]

                    new_lp = 0.0
                    entropy = 0.0

                    dist_req = Categorical(logits=req_logits)
                    new_lp = new_lp + dist_req.log_prob(torch.tensor(action.req_idx, device=self.device))
                    entropy = entropy + dist_req.entropy()

                    has_prev = bool(meta["has_prev"])
                    prev_RU = int(meta["prev_RU"])
                    prev_DU = int(meta["prev_DU"])
                    prev_CU = int(meta["prev_CU"])

                    ru_valid = state.ru_mask & state.ru_valid_mask
                    du_valid = state.du_mask & state.du_valid_mask
                    cu_valid = state.cu_mask & state.cu_valid_mask

                    alt_ru_available = False
                    if has_prev and 0 <= prev_RU < ru_valid.numel():
                        tmp = ru_valid.clone()
                        tmp[prev_RU] = False
                        alt_ru_available = bool(tmp.any().item())

                    if has_prev and alt_ru_available:
                        dist_ho = Categorical(logits=self.policy.score_handover(req_vec))
                        new_lp = new_lp + dist_ho.log_prob(torch.tensor(action.handover_flag, device=self.device))
                        entropy = entropy + dist_ho.entropy()

                    if has_prev and action.handover_flag == 0:
                        # keep mapping -> no node selection logprob
                        pass
                    else:
                        if has_prev and action.handover_flag == 1 and 0 <= prev_RU < ru_valid.numel():
                            ru_valid = ru_valid.clone()
                            ru_valid[prev_RU] = False

                        ru_idx_pool, ru_logits = self.policy.score_nodes(node_emb, req_vec, ru_valid)
                        du_idx_pool, du_logits = self.policy.score_nodes(node_emb, req_vec, du_valid)
                        cu_idx_pool, cu_logits = self.policy.score_nodes(node_emb, req_vec, cu_valid)

                        dist_ru = Categorical(logits=ru_logits)
                        dist_du = Categorical(logits=du_logits)
                        dist_cu = Categorical(logits=cu_logits)

                        # action indices are stored as global node indices, so convert to local index in pool
                        def _pool_pos(pool, global_idx):
                            pos = (pool == global_idx).nonzero(as_tuple=False)
                            if pos.numel() == 0:
                                return None
                            return int(pos.item())

                        ru_pos = _pool_pos(ru_idx_pool, action.ru_idx)
                        du_pos = _pool_pos(du_idx_pool, action.du_idx)
                        cu_pos = _pool_pos(cu_idx_pool, action.cu_idx)
                        if ru_pos is None or du_pos is None or cu_pos is None:
                            continue

                        new_lp = new_lp + dist_ru.log_prob(torch.tensor(ru_pos, device=self.device))
                        new_lp = new_lp + dist_du.log_prob(torch.tensor(du_pos, device=self.device))
                        new_lp = new_lp + dist_cu.log_prob(torch.tensor(cu_pos, device=self.device))

                        entropy = entropy + dist_ru.entropy() + dist_du.entropy() + dist_cu.entropy()

                    dist_rb = Categorical(logits=self.policy.score_rb(req_vec))
                    dist_pw = Categorical(logits=self.policy.score_power(req_vec))
                    new_lp = new_lp + dist_rb.log_prob(torch.tensor(action.rb_idx, device=self.device))
                    new_lp = new_lp + dist_pw.log_prob(torch.tensor(action.power_idx, device=self.device))
                    entropy = entropy + dist_rb.entropy() + dist_pw.entropy()

                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                    policy_loss = -torch.min(surr1, surr2)

                    v_pred = v_pred.squeeze(-1)
                    v_clip = v_old + torch.clamp(v_pred - v_old, -self.clip_ratio, self.clip_ratio)
                    v_loss1 = (ret - v_pred).pow(2)
                    v_loss2 = (ret - v_clip).pow(2)
                    value_loss = torch.max(v_loss1, v_loss2)

                    approx_kl = (old_lp - new_lp).detach()

                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    entropies.append(entropy)
                    kls.append(approx_kl)

                if len(policy_losses) == 0:
                    continue

                loss = (
                    torch.stack(policy_losses).mean()
                    + self.value_coef * torch.stack(value_losses).mean()
                    - self.entropy_coef * torch.stack(entropies).mean()
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                if len(kls) > 0:
                    approx_kls.append(torch.stack(kls).mean().item())

            if len(approx_kls) > 0 and float(np.mean(approx_kls)) > 1.5 * self.target_kl:
                break


# =========================================================
# Train / Evaluate
# =========================================================
def train_agent(env, agent, results_dir, max_episode=MAX_EPISODE, update_per_steps=STEPS_PER_UPDATE):
    os.makedirs(results_dir, exist_ok=True)

    files = {
        "reward": open(os.path.join(results_dir, "reward.txt"), "a"),
        "accept": open(os.path.join(results_dir, "accept.txt"), "a"),
        "throughput": open(os.path.join(results_dir, "throughput.txt"), "a"),
        "latency": open(os.path.join(results_dir, "latency.txt"), "a"),
        "handover": open(os.path.join(results_dir, "handover.txt"), "a"),
        "pingpong": open(os.path.join(results_dir, "pingpong.txt"), "a"),
        "acc_term": open(os.path.join(results_dir, "acc_term.txt"), "a"),
        "thr_term": open(os.path.join(results_dir, "thr_term.txt"), "a"),
        "lat_term": open(os.path.join(results_dir, "lat_term.txt"), "a"),
        "ho_term": open(os.path.join(results_dir, "handover_term.txt"), "a"),
        "num_ues": open(os.path.join(results_dir, "num_ues.txt"), "a"),
        "time": open(os.path.join(results_dir, "time.txt"), "a"),
    }

    buf = RolloutBuffer(states=[], actions=[], logprobs=[], rewards=[], values=[], next_values=[], masks=[])
    steps_collected = 0

    try:
        for ep in range(1, max_episode + 1):
            state = env.reset_env()

            ep_reward = ep_accept = ep_thr = ep_lat = ep_ho = ep_pingpong = 0.0
            ep_acc_term = ep_thr_term = ep_lat_term = ep_ho_term = 0.0

            done = False
            start_time = time.time()

            while not done:
                cached_state, cached_action, logp, value = agent.select_action(state)
                if cached_action is None:
                    break

                next_state, reward, done, info = env.step(cached_action.env_action)
                next_graph_state = agent.graph_builder.build_graph_state(next_state)
                v_next = agent.get_value(next_graph_state) if not done else 0.0

                buf.states.append(cached_state)
                buf.actions.append(cached_action)
                buf.logprobs.append(float(logp) if logp is not None else 0.0)
                buf.values.append(float(value))
                buf.next_values.append(float(v_next))
                buf.rewards.append(float(reward))
                buf.masks.append(0 if done else 1)

                ep_reward += float(reward)

                if isinstance(info, dict) and info.get("success", False):
                    ep_accept += 1
                    ep_thr += float(info.get("throughput_UE", 0.0))
                    ep_lat += float(info.get("latency_UE", 0.0))
                    ep_ho += float(info.get("handover", 0.0))
                    ep_pingpong += float(info.get("pingpong", 0.0))
                    ep_acc_term += float(info.get("acc_term", 0.0))
                    ep_thr_term += float(info.get("thr_term", 0.0))
                    ep_lat_term += float(info.get("lat_term", 0.0))
                    ep_ho_term += float(info.get("HO_term", 0.0))

                state = next_state
                steps_collected += 1

                if steps_collected >= update_per_steps:
                    agent.update(buf, episode_idx=ep)
                    buf = RolloutBuffer(states=[], actions=[], logprobs=[], rewards=[], values=[], next_values=[], masks=[])
                    steps_collected = 0

            elapsed = time.time() - start_time
            num_ues = len(env.UE_manager.UE_requests)

            files["reward"].write(f"{ep_reward}\n")
            files["accept"].write(f"{ep_accept}\n")
            files["throughput"].write(f"{ep_thr}\n")
            files["latency"].write(f"{ep_lat}\n")
            files["handover"].write(f"{ep_ho}\n")
            files["pingpong"].write(f"{ep_pingpong}\n")
            files["acc_term"].write(f"{ep_acc_term}\n")
            files["thr_term"].write(f"{ep_thr_term}\n")
            files["lat_term"].write(f"{ep_lat_term}\n")
            files["ho_term"].write(f"{ep_ho_term}\n")
            files["num_ues"].write(f"{num_ues}\n")
            files["time"].write(f"{elapsed:.2f}\n")

            for f in files.values():
                f.flush()

            avg_thr = ep_thr / max(ep_accept, 1) / 1e6
            avg_lat = ep_lat / max(ep_accept, 1) * 1e3
            ho_rate = ep_ho / max(ep_accept, 1)
            pp_rate = ep_pingpong / max(ep_accept, 1)

            print(
                f"[Episode {ep:05d}/{max_episode}] "
                f"Reward={ep_reward:8.2f} | "
                f"Accept={ep_accept:3.0f}/{num_ues:3.0f} | "
                f"Thr={ep_thr/1e6:7.2f}({avg_thr:6.2f})Mb | "
                f"Lat={ep_lat*1e3:7.2f}({avg_lat:6.2f})ms | "
                f"HO={ep_ho:3.0f}({ho_rate:.2f}) | "
                f"PP={ep_pingpong:3.0f}({pp_rate:.2f})"
            )

        if len(buf.rewards) > 0:
            agent.update(buf, episode_idx=max_episode)

        return agent
    finally:
        for f in files.values():
            f.close()


def evaluate_agent(env, agent, render=False):
    state = env.reset_env()

    ep_reward = ep_accept = ep_thr = ep_lat = ep_ho = ep_pingpong = 0.0
    ep_acc_term = ep_thr_term = ep_lat_term = ep_ho_term = 0.0

    done = False
    start_time = time.time()

    while not done:
        cached_state, cached_action, logp, value = agent.select_action(state)
        if cached_action is None:
            break

        next_state, reward, done, info = env.step(cached_action.env_action)
        ep_reward += float(reward)

        if isinstance(info, dict) and info.get("success", False):
            ep_accept += 1
            ep_thr += float(info.get("throughput_UE", 0.0))
            ep_lat += float(info.get("latency_UE", 0.0))
            ep_ho += float(info.get("handover", 0.0))
            ep_pingpong += float(info.get("pingpong", 0.0))
            ep_acc_term += float(info.get("acc_term", 0.0))
            ep_thr_term += float(info.get("thr_term", 0.0))
            ep_lat_term += float(info.get("lat_term", 0.0))
            ep_ho_term += float(info.get("HO_term", 0.0))

        state = next_state
        if render:
            print(info)

    elapsed = time.time() - start_time
    num_ues = len(env.UE_manager.UE_requests)

    return (
        ep_reward,
        ep_accept,
        ep_thr,
        ep_lat,
        ep_ho,
        ep_pingpong,
        ep_acc_term,
        ep_thr_term,
        ep_lat_term,
        ep_ho_term,
        elapsed,
        num_ues,
    )


# =========================================================
# Checkpoint utils
# =========================================================
def save_checkpoint(agent, path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "policy_state": agent.policy.state_dict(),
            "optimizer_state": agent.optimizer.state_dict(),
            "scaler_state": agent.scaler.state_dict() if agent.scaler is not None else None,
            "entropy_coef": float(agent.entropy_coef),
            "meta": {
                "node_in": getattr(agent, "node_in", None),
                "edge_in": getattr(agent, "edge_in", None),
                "request_in": getattr(agent, "request_in", None),
                "hidden_dim": getattr(agent, "hidden_dim", None),
            },
        }
        torch.save(checkpoint, path)
        print(f"Đã lưu checkpoint tại: {path}")
    except Exception as e:
        print(f"Lỗi khi lưu checkpoint: {e}")


def load_partial_policy_state(policy: nn.Module, state_dict: dict, strict: bool = False):
    current_sd = policy.state_dict()
    filtered = {}
    skipped = []

    for k, v in state_dict.items():
        if k in current_sd and current_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    current_sd.update(filtered)
    policy.load_state_dict(current_sd, strict=strict)
    return filtered, skipped


def load_checkpoint(agent, path, strict=False, partial=True):
    if not os.path.exists(path):
        print(f"Không tìm thấy checkpoint tại: {path}")
        return False

    try:
        ckpt = torch.load(path, map_location=agent.device)
        policy_state = ckpt.get("policy_state", {})

        if partial:
            loaded, skipped = load_partial_policy_state(agent.policy, policy_state, strict=False)
            print(f"Load partial checkpoint từ: {path}")
            print(f" - Tensor load được: {len(loaded)}")
            print(f" - Tensor bỏ qua: {len(skipped)}")
        else:
            agent.policy.load_state_dict(policy_state, strict=strict)
            print(f"Đã load checkpoint từ: {path}")

        if "optimizer_state" in ckpt:
            try:
                agent.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                pass

        if ckpt.get("scaler_state", None) is not None and agent.scaler is not None:
            try:
                agent.scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                pass

        agent.entropy_coef = float(ckpt.get("entropy_coef", agent.entropy_coef))
        return True
    except Exception as e:
        print(f"Lỗi khi load checkpoint từ {path}: {e}")
        return False