from __future__ import annotations

import os
import time
import math
import copy
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

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


def safe_reset_env(env):
    if hasattr(env, "reset_env") and callable(getattr(env, "reset_env")):
        return env.reset_env()

    if hasattr(env, "resource_manager") and hasattr(env, "UE_manager") and hasattr(env, "get_state"):
        env.resource_manager.reset()
        env.UE_manager = env.UE_manager.__class__(env.resource_manager.coordinates_RU)
        num_ues = int(getattr(env, "num_UEs", 0))
        if num_ues > 0:
            _ = env.UE_manager.add_UEs_requests(num_ues)
        return env.get_state()

    raise AttributeError(
        "Environment must provide reset_env(), or at least resource_manager / UE_manager / get_state()."
    )


# ===================================================
@dataclass
class CachedState:
    node_feats: torch.Tensor
    edge_feats: torch.Tensor
    request_feats: torch.Tensor
    orig_ids: List[int]
    ru_mask: torch.Tensor
    du_mask: torch.Tensor
    cu_mask: torch.Tensor
    raw_state: dict


@dataclass
class CachedAction:
    env_action: Tuple[int, int, int, int, int, int, float]
    req_idx: int

    has_prev_alloc: bool
    mode_sampled: bool
    node_sampled: bool

    handover_flag: int

    # local indices for env.step()
    ru_local: int
    du_local: int
    cu_local: int

    # global indices in the full graph space
    ru_global: int
    du_global: int
    cu_global: int

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

    def _infer_sizes(self, state):
        RAN = state.get("RAN", {})
        ru = np.asarray(RAN.get("RU_power_remaining", []), dtype=float)
        du = np.asarray(RAN.get("DU_remaining", []), dtype=float)
        cu = np.asarray(RAN.get("CU_remaining", []), dtype=float)
        lru = np.asarray(RAN.get("link_bw_ru_du_bps", []), dtype=float)
        ldu = np.asarray(RAN.get("link_bw_du_cu_bps", []), dtype=float)
        return len(ru), len(du), len(cu), ru, du, cu, lru, ldu

    def build_node_masks(self, state):
        num_RUs, num_DUs, num_CUs, *_ = self._infer_sizes(state)
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
        total_nodes = num_RUs + num_DUs + num_CUs

        if total_nodes == 0:
            return torch.zeros((0, self.node_in), dtype=torch.float32, device=self.device)

        ru_scale = max(float(np.max(RU)) if RU.size else 1.0, 1.0)
        du_scale = max(float(np.max(DU)) if DU.size else 1.0, 1.0)
        cu_scale = max(float(np.max(CU)) if CU.size else 1.0, 1.0)

        RU_n = RU / (ru_scale + 1e-9) if RU.size else np.zeros((0,), dtype=np.float32)
        DU_n = DU / (du_scale + 1e-9) if DU.size else np.zeros((0,), dtype=np.float32)
        CU_n = CU / (cu_scale + 1e-9) if CU.size else np.zeros((0,), dtype=np.float32)

        remain = np.concatenate([RU_n, DU_n, CU_n], axis=0).astype(np.float32)

        link_ru_du = np.asarray(state.get("RAN", {}).get("link_bw_ru_du_bps", []), dtype=float)
        link_du_cu = np.asarray(state.get("RAN", {}).get("link_bw_du_cu_bps", []), dtype=float)

        degree = np.zeros((total_nodes,), dtype=np.float32)
        if link_ru_du.size > 0:
            for i in range(num_RUs):
                degree[i] += float(np.sum(link_ru_du[i] > 0))
            for j in range(num_DUs):
                degree[num_RUs + j] += float(np.sum(link_ru_du[:, j] > 0))
        if link_du_cu.size > 0:
            for j in range(num_DUs):
                degree[num_RUs + j] += float(np.sum(link_du_cu[j] > 0))
            for k in range(num_CUs):
                degree[num_RUs + num_DUs + k] += float(np.sum(link_du_cu[:, k] > 0))

        deg_scale = max(float(total_nodes - 1), 1.0)
        degree = degree / deg_scale

        remain_t = torch.tensor(remain, dtype=torch.float32, device=self.device)
        if remain_t.numel() > 1:
            remain_t = (remain_t - remain_t.mean()) / (remain_t.std() + 1e-6)
        else:
            remain_t = remain_t * 0.0

        degree_t = torch.tensor(degree, dtype=torch.float32, device=self.device).unsqueeze(-1)
        type_oh = self._node_type_onehot(num_RUs, num_DUs, num_CUs)

        node_feats = torch.cat([remain_t.unsqueeze(-1), degree_t, type_oh], dim=-1)
        return node_feats

    def build_edge_features(self, state):
        num_RUs, num_DUs, num_CUs, _, _, _, link_ru_du, link_du_cu = self._infer_sizes(state)
        total_nodes = num_RUs + num_DUs + num_CUs

        if total_nodes == 0:
            return torch.zeros((0, 0, 1), dtype=torch.float32, device=self.device)

        bw_scale = max(
            float(np.max(link_ru_du)) if link_ru_du.size else 1.0,
            float(np.max(link_du_cu)) if link_du_cu.size else 1.0,
            1.0,
        )

        adj = np.zeros((total_nodes, total_nodes), dtype=np.float32)

        if link_ru_du.size > 0:
            nRU = min(num_RUs, link_ru_du.shape[0])
            nDU = min(num_DUs, link_ru_du.shape[1])
            for i in range(nRU):
                for j in range(nDU):
                    bw = float(link_ru_du[i, j]) / (bw_scale + 1e-9)
                    adj[i, num_RUs + j] = bw
                    adj[num_RUs + j, i] = bw

        if link_du_cu.size > 0:
            nDU = min(num_DUs, link_du_cu.shape[0])
            nCU = min(num_CUs, link_du_cu.shape[1])
            for j in range(nDU):
                for k in range(nCU):
                    bw = float(link_du_cu[j, k]) / (bw_scale + 1e-9)
                    adj[num_RUs + j, num_RUs + num_DUs + k] = bw
                    adj[num_RUs + num_DUs + k, num_RUs + j] = bw

        np.fill_diagonal(adj, 1.0)
        return torch.from_numpy(adj).to(self.device).unsqueeze(-1)

    def build_request_features(self, state):
        UE_requests_snapshot = state.get("UE_requests", {})
        if not UE_requests_snapshot:
            return torch.zeros((0, self.request_in), dtype=torch.float32, device=self.device), []

        active_items = []
        for ue_id, UE in sorted(UE_requests_snapshot.items(), key=lambda x: x[0]):
            active = int(UE.get("status", {}).get("active", UE.get("active", 0)))
            if active == 1:
                active_items.append((ue_id, UE))

        if len(active_items) == 0:
            return torch.zeros((0, self.request_in), dtype=torch.float32, device=self.device), []

        r_min_list = []
        delay_list = []
        pkt_list = []
        cyc_list = []
        lam_list = []
        gain_best_list = []
        dist_best_list = []

        for _, UE in active_items:
            r_min_list.append(float(UE.get("R_min", 1.0)))
            delay_list.append(float(UE.get("delay", 1e-3)))
            pkt_list.append(float(UE.get("packet_size_bits", 1.0)))
            cyc_list.append(float(UE.get("cycles_per_packet", 1.0)))
            lam_list.append(float(UE.get("lambda_default_pps", 1.0)))

            gains = np.asarray(UE.get("gain", []), dtype=float)
            dists = np.asarray(UE.get("distances_RU_UE", []), dtype=float)
            gain_best_list.append(float(np.max(gains)) if gains.size else 0.0)
            dist_best_list.append(float(np.min(dists)) if dists.size else 0.0)

        r_scale = max(max(r_min_list), 1.0)
        d_scale = max(max(delay_list), 1e-6)
        p_scale = max(max(pkt_list), 1.0)
        c_scale = max(max(cyc_list), 1.0)
        l_scale = max(max(lam_list), 1.0)
        g_scale = max(max(gain_best_list), 1.0)
        dist_scale = max(max(dist_best_list), 1.0)

        num_RUs, num_DUs, num_CUs, *_ = self._infer_sizes(state)

        feats = []
        ids = []

        for ue_id, UE in active_items:
            prev = UE.get("allocation", {})
            has_prev = 1.0 if prev.get("RU") is not None else 0.0

            gains = np.asarray(UE.get("gain", []), dtype=float)
            dists = np.asarray(UE.get("distances_RU_UE", []), dtype=float)

            prev_RU = prev.get("RU")
            prev_DU = prev.get("DU")
            prev_CU = prev.get("CU")

            prev_RU = -1 if prev_RU is None else int(prev_RU)
            prev_DU = -1 if prev_DU is None else int(prev_DU)
            prev_CU = -1 if prev_CU is None else int(prev_CU)

            prev_RB = float(prev.get("num_RB_alloc") or 0.0)
            prev_power = float(prev.get("power_alloc") or 0.0)

            feats.append([
                float(UE.get("R_min", 1.0)) / (r_scale + 1e-9),
                float(UE.get("delay", 1e-3)) / (d_scale + 1e-9),
                float(UE.get("packet_size_bits", 1.0)) / (p_scale + 1e-9),
                float(UE.get("cycles_per_packet", 1.0)) / (c_scale + 1e-9),
                float(UE.get("lambda_default_pps", 1.0)) / (l_scale + 1e-9),

                float(UE.get("eta_slice", 0.0)),
                float(UE.get("weight_accept", 1.0)),
                float(UE.get("weight_throughput", 1.0)),
                float(UE.get("weight_latency", 1.0)),
                float(UE.get("weight_handover", 1.0)),

                (float(np.max(gains)) if gains.size else 0.0) / (g_scale + 1e-9),
                (float(np.mean(gains)) if gains.size else 0.0) / (g_scale + 1e-9),
                (float(np.min(dists)) if dists.size else 0.0) / (dist_scale + 1e-9),

                has_prev,
                (prev_RU / max(num_RUs - 1, 1)) if prev_RU >= 0 else 0.0,
                (prev_DU / max(num_DUs - 1, 1)) if prev_DU >= 0 else 0.0,
                (prev_CU / max(num_CUs - 1, 1)) if prev_CU >= 0 else 0.0,
                prev_RB / max(max_RBs_per_UE, 1),
                prev_power / (max(P_ib_sk_val) if len(P_ib_sk_val) > 0 else 1.0),
                float(UE.get("handover_count", 0.0)) / 10.0,
                float(UE.get("pingpong", 0.0)) / 10.0,
            ])
            ids.append(int(ue_id))

        x = torch.tensor(np.asarray(feats, dtype=np.float32), device=self.device)
        return x, ids

    def build_graph_state(self, state):
        node_feats = self.build_node_features(state)
        edge_feats = self.build_edge_features(state)
        req_feats, ids = self.build_request_features(state)
        ru_mask, du_mask, cu_mask = self.build_node_masks(state)

        return CachedState(
            node_feats=node_feats,
            edge_feats=edge_feats,
            request_feats=req_feats,
            orig_ids=ids,
            ru_mask=ru_mask,
            du_mask=du_mask,
            cu_mask=cu_mask,
            raw_state=copy.deepcopy(state),
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
        if node_feats.numel() == 0:
            graph_emb = torch.zeros((self.hidden_dim,), device=node_feats.device)
            return node_feats, graph_emb

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
            nn.Linear(int(req_in), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
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

        self.graph_encoder = GraphSAGEEncoder(
            node_in=node_in,
            edge_in=edge_in,
            hidden_dim=hidden_dim,
            num_layers=num_graph_layers,
        )

        self.req_encoder = MLPEncode(req_in=req_in, hidden_dim=hidden_dim)

        self.req_head = nn.Linear(hidden_dim * 2, 1)
        self.mode_head = nn.Linear(hidden_dim * 2, 2)

        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.rb_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_RBs_per_UE),
        )

        self.power_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_power_levels),
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                orth_init(m, math.sqrt(2))

        orth_init(self.req_head, 0.01)
        orth_init(self.mode_head, 0.01)

    def encode(self, graph_state):
        node_emb, graph_emb = self.graph_encoder(graph_state.node_feats, graph_state.edge_feats)

        if graph_state.request_feats.shape[0] == 0:
            req_emb = torch.zeros((0, self.hidden_dim), device=graph_state.node_feats.device)
            req_logits = torch.zeros((0,), device=graph_state.node_feats.device)
            value = self.value_head(torch.cat([graph_emb, graph_emb], dim=-1)).squeeze(-1)
            return node_emb, graph_emb, req_emb, req_logits, value

        req_emb = self.req_encoder(graph_state.request_feats)
        req_ctx = graph_emb.unsqueeze(0).expand(req_emb.shape[0], -1)
        req_logits = self.req_head(torch.cat([req_emb, req_ctx], dim=-1)).squeeze(-1)

        value_in = torch.cat([graph_emb, req_emb.mean(dim=0)], dim=-1)
        value = self.value_head(value_in).squeeze(-1)
        return node_emb, graph_emb, req_emb, req_logits, value

    def score_mode(self, req_vec, graph_emb):
        return self.mode_head(torch.cat([req_vec, graph_emb], dim=-1))

    def score_nodes(self, req_vec, node_emb, graph_emb):
        N = node_emb.shape[0]
        ctx = graph_emb.unsqueeze(0).expand(N, -1)
        req_ctx = req_vec.unsqueeze(0).expand(N, -1)
        return self.node_decoder(torch.cat([node_emb, req_ctx, ctx], dim=-1)).squeeze(-1)

    def score_rb(self, req_vec, graph_emb):
        return self.rb_head(torch.cat([req_vec, graph_emb], dim=-1))

    def score_power(self, req_vec, graph_emb, chosen_ru_emb):
        return self.power_head(torch.cat([req_vec, graph_emb, chosen_ru_emb], dim=-1))


# =========================================================
# PPO Agent
# =========================================================
class PPOAgent:
    def __init__(
        self,
        learning_rate,
        num_RUs=None,
        num_DUs=None,
        num_CUs=None,
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

        self.node_in = 5
        self.edge_in = 1
        self.request_in = 21
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

        self.scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

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

    def _get_ue_raw(self, raw_state, ue_id):
        return raw_state["UE_requests"][int(ue_id)]

    def _effective_remaining(self, raw_state, ue_id):
        RAN = raw_state["RAN"]

        rb_rem = int(RAN["RB_remaining"])
        ru_rem = np.array(RAN["RU_power_remaining"], dtype=float).copy()
        du_rem = np.array(RAN["DU_remaining"], dtype=float).copy()
        cu_rem = np.array(RAN["CU_remaining"], dtype=float).copy()

        UE = self._get_ue_raw(raw_state, ue_id)
        prev = UE.get("allocation", {})

        if prev.get("RU") is not None:
            rb_rem += int(prev.get("num_RB_alloc", 0))
            ru_rem[int(prev["RU"])] += float(prev.get("power_alloc", 0.0))
            du_rem[int(prev["DU"])] += float(prev.get("cpu_DU_req", 0.0))
            cu_rem[int(prev["CU"])] += float(prev.get("cpu_CU_req", 0.0))

        return rb_rem, ru_rem, du_rem, cu_rem

    def _masked_categorical(self, logits, valid_mask=None):
        if logits.numel() == 0:
            raise ValueError("Empty logits.")

        if valid_mask is None:
            return Categorical(logits=logits)

        if valid_mask.sum() == 0:
            raise RuntimeError("No valid action after applying mask.")

        masked_logits = logits.clone()
        masked_logits[~valid_mask] = -1e9

        return Categorical(logits=masked_logits)

    def _sample_from_logits(self, logits, valid_mask=None):
        dist = self._masked_categorical(logits, valid_mask)
        idx_t = dist.sample()
        return int(idx_t.item()), dist.log_prob(idx_t), dist.entropy()

    def _local_to_global(self, state, group_name, local_idx):
        num_RUs, num_DUs, num_CUs, *_ = self.graph_builder._infer_sizes(state)
        if group_name == "RU":
            return int(local_idx)
        elif group_name == "DU":
            return int(num_RUs + local_idx)
        elif group_name == "CU":
            return int(num_RUs + num_DUs + local_idx)
        raise ValueError(f"Unknown group_name={group_name}")

    def _build_valid_masks(self, raw_state, ue_id, exclude_prev_ru_local=None):
        rb_rem, ru_rem, du_rem, cu_rem = self._effective_remaining(raw_state, ue_id)

        RAN = raw_state["RAN"]
        link_ru_du = np.asarray(RAN["link_bw_ru_du_bps"], dtype=float)
        link_du_cu = np.asarray(RAN["link_bw_du_cu_bps"], dtype=float)

        if link_ru_du.size > 0:
            ru_connect = (link_ru_du.sum(axis=1) > 0)
            du_connect_from_ru = (link_ru_du.sum(axis=0) > 0)
        else:
            ru_connect = np.ones_like(ru_rem, dtype=bool)
            du_connect_from_ru = np.ones_like(du_rem, dtype=bool)

        if link_du_cu.size > 0:
            du_connect_from_cu = (link_du_cu.sum(axis=1) > 0)
            cu_connect = (link_du_cu.sum(axis=0) > 0)
        else:
            du_connect_from_cu = np.ones_like(du_rem, dtype=bool)
            cu_connect = np.ones_like(cu_rem, dtype=bool)

        min_power = float(np.min(self.power_levels))
        ru_valid = ((ru_rem + 1e-12 >= min_power)& ru_connect)
        du_valid = (du_rem > 1e-12) & du_connect_from_ru & du_connect_from_cu
        cu_valid = (cu_rem > 1e-12) & cu_connect

        if exclude_prev_ru_local is not None and 0 <= int(exclude_prev_ru_local) < len(ru_valid):
            if int(ru_valid.sum()) > 1:
                ru_valid[int(exclude_prev_ru_local)] = False

        if len(ru_valid) > 0 and ru_valid.sum() == 0:
            ru_valid[:] = True
        if len(du_valid) > 0 and du_valid.sum() == 0:
            du_valid[:] = True
        if len(cu_valid) > 0 and cu_valid.sum() == 0:
            cu_valid[:] = True

        # rb_max = max(1, min(self.policy.max_RBs_per_UE, rb_rem))
        # rb_valid = torch.zeros(self.policy.max_RBs_per_UE, dtype=torch.bool, device=self.device)
        # rb_valid[:rb_max] = True
        # =====================================================
        # RB mask theo từng loại slice
        # =====================================================
        UE = self._get_ue_raw(
            raw_state,
            ue_id,
        )

        slice_max_RBs = int(
            UE.get(
                "max_RBs",
                self.policy.max_RBs_per_UE,
            )
        )

        rb_max = max(
            1,
            min(
                self.policy.max_RBs_per_UE,
                slice_max_RBs,
                rb_rem,
            ),
        )

        rb_valid = torch.zeros(
            self.policy.max_RBs_per_UE,
            dtype=torch.bool,
            device=self.device,
        )

        rb_valid[:rb_max] = True

        return (
            torch.tensor(ru_valid, dtype=torch.bool, device=self.device),
            torch.tensor(du_valid, dtype=torch.bool, device=self.device),
            torch.tensor(cu_valid, dtype=torch.bool, device=self.device),
            rb_valid,
            ru_rem,
        )

    def compute_GAE(self, rewards, values, next_values, masks):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_values[t] * masks[t] - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * masks[t] * last_gae
            advantages[t] = last_gae

        returns = advantages + np.asarray(values, dtype=np.float32)
        return returns, advantages

    def select_action(self, state):
        graph_state = self.graph_builder.build_graph_state(state)

        if graph_state.request_feats.shape[0] == 0:
            return graph_state, None, None, 0.0

        raw_state = graph_state.raw_state

        self.policy.eval()
        with torch.no_grad():
            node_emb, graph_emb, req_emb, req_logits, value = self.policy.encode(graph_state)

            req_dist = Categorical(logits=req_logits)
            req_idx_t = req_dist.sample()
            req_idx = int(req_idx_t.item())
            logp = req_dist.log_prob(req_idx_t)

            ue_id = int(graph_state.orig_ids[req_idx])
            UE = self._get_ue_raw(raw_state, ue_id)
            prev = UE.get("allocation", {})
            has_prev = prev.get("RU") is not None

            req_vec = req_emb[req_idx]

            prev_ru_local = (
                int(prev["RU"])
                if has_prev and prev.get("RU") is not None
                else None
            )

            # =====================================================
            # Build resource masks
            # Không loại RU cũ ở bước này.
            # RU cũ chỉ bị loại nếu agent thực sự chọn handover.
            # =====================================================
            ru_valid, du_valid, cu_valid, rb_valid, ru_rem = (
                self._build_valid_masks(
                    raw_state,
                    ue_id,
                    exclude_prev_ru_local=None,
                )
            )

            # =====================================================
            # Sample handover mode
            # =====================================================
            mode_sampled = False

            if has_prev and int(ru_valid.sum().item()) > 1:
                mode_dist = Categorical(
                    logits=self.policy.score_mode(
                        req_vec,
                        graph_emb,
                    )
                )

                mode_t = mode_dist.sample()
                handover_flag = int(mode_t.item())

                logp = logp + mode_dist.log_prob(mode_t)
                mode_sampled = True
            else:
                handover_flag = 0

            # keep mapping
            if has_prev and handover_flag == 0:
                node_sampled = False

                ru_local = int(prev["RU"])
                du_local = int(prev["DU"])
                cu_local = int(prev["CU"])

                ru_global = self._local_to_global(raw_state, "RU", ru_local)
                du_global = self._local_to_global(raw_state, "DU", du_local)
                cu_global = self._local_to_global(raw_state, "CU", cu_local)

                chosen_ru_emb = node_emb[ru_global]

                rb_logits = self.policy.score_rb(req_vec, graph_emb)
                rb_idx, lp_rb, _ = self._sample_from_logits(rb_logits, rb_valid)
                logp = logp + lp_rb

                power_logits = self.policy.score_power(req_vec, graph_emb, chosen_ru_emb)
                power_available = float(ru_rem[ru_global])
                power_mask = torch.tensor(
                    [float(p) <= power_available + 1e-12 for p in self.power_levels],
                    dtype=torch.bool,
                    device=self.device,
                )
                if power_mask.sum() == 0:
                    raise RuntimeError(
                        f"No affordable power level: "
                        f"UE={ue_id}, RU={ru_local}, "
                        f"available_power={power_available:.6f}, "
                        f"min_power={min(self.power_levels):.6f}"
                    )
                power_idx, lp_pw, _ = self._sample_from_logits(power_logits, power_mask)
                logp = logp + lp_pw

            # initial attach or handover
            else:
                node_sampled = True

                node_logits = self.policy.score_nodes(req_vec, node_emb, graph_emb)

                ru_logits = node_logits[graph_state.ru_mask]
                du_logits = node_logits[graph_state.du_mask]
                cu_logits = node_logits[graph_state.cu_mask]

                # if has_prev and handover_flag == 1 and ru_logits.numel() > 1:
                #     prev_ru_local = int(prev["RU"])
                #     valid_ru_local = torch.ones_like(ru_logits, dtype=torch.bool)
                #     valid_ru_local[prev_ru_local] = False
                # else:
                #     valid_ru_local = None

                # ru_local, lp_ru, _ = self._sample_from_logits(ru_logits, valid_ru_local)
                # du_local, lp_du, _ = self._sample_from_logits(du_logits)
                # cu_local, lp_cu, _ = self._sample_from_logits(cu_logits)
                
                # =====================================================
                # Áp dụng mask tài nguyên khi chọn RU / DU / CU
                # =====================================================
                valid_ru_local = ru_valid.clone()

                # Nếu thực hiện handover thì không được chọn lại RU cũ
                if has_prev and handover_flag == 1:
                    prev_ru_local = int(prev["RU"])

                    if 0 <= prev_ru_local < valid_ru_local.numel():
                        valid_ru_local[prev_ru_local] = False

                # Trường hợp không còn RU hợp lệ sau khi loại RU cũ:
                # tránh lỗi phân phối rỗng, cho phép chọn các RU khác RU cũ.
                if valid_ru_local.sum() == 0:
                    valid_ru_local = torch.ones_like(
                        ru_logits,
                        dtype=torch.bool,
                        device=self.device,
                    )

                    if has_prev and handover_flag == 1:
                        valid_ru_local[int(prev["RU"])] = False
                if valid_ru_local.sum() == 0:
                    raise RuntimeError(
                        f"No valid RU after masking: "
                        f"UE={ue_id}, "
                        f"handover_flag={handover_flag}, "
                        f"prev_RU={prev.get('RU')}"
                    )
                # Sample node với mask tài nguyên
                ru_local, lp_ru, _ = self._sample_from_logits(
                    ru_logits,
                    valid_ru_local,
                )

                du_local, lp_du, _ = self._sample_from_logits(
                    du_logits,
                    du_valid,
                )

                cu_local, lp_cu, _ = self._sample_from_logits(
                    cu_logits,
                    cu_valid,
                )
                ru_global = self._local_to_global(raw_state, "RU", ru_local)
                du_global = self._local_to_global(raw_state, "DU", du_local)
                cu_global = self._local_to_global(raw_state, "CU", cu_local)

                logp = logp + lp_ru + lp_du + lp_cu

                chosen_ru_emb = node_emb[ru_global]

                rb_logits = self.policy.score_rb(req_vec, graph_emb)
                rb_idx, lp_rb, _ = self._sample_from_logits(rb_logits, rb_valid)
                logp = logp + lp_rb

                power_logits = self.policy.score_power(req_vec, graph_emb, chosen_ru_emb)
                power_available = float(ru_rem[ru_global])
                power_mask = torch.tensor(
                    [float(p) <= power_available + 1e-12 for p in self.power_levels],
                    dtype=torch.bool,
                    device=self.device,
                )
                if power_mask.sum() == 0:
                    power_mask[:] = True
                power_idx, lp_pw, _ = self._sample_from_logits(power_logits, power_mask)
                logp = logp + lp_pw

            num_RB_alloc = int(rb_idx + 1)
            power_alloc = float(self.power_levels[power_idx])

            env_action = (
                ue_id,
                int(handover_flag),
                int(ru_local),
                int(du_local),
                int(cu_local),
                int(num_RB_alloc),
                float(power_alloc),
            )

            cached_action = CachedAction(
                env_action=env_action,
                req_idx=req_idx,
                has_prev_alloc=bool(has_prev),
                mode_sampled=bool(mode_sampled),
                node_sampled=bool(node_sampled),
                handover_flag=int(handover_flag),
                ru_local=int(ru_local),
                du_local=int(du_local),
                cu_local=int(cu_local),
                ru_global=int(ru_global),
                du_global=int(du_global),
                cu_global=int(cu_global),
                rb_idx=int(rb_idx),
                power_idx=int(power_idx),
            )

            return graph_state, cached_action, float(logp.item()), float(value.item())

    def get_value(self, graph_state):
        self.policy.eval()
        with torch.no_grad():
            _, _, _, _, value = self.policy.encode(graph_state)
        return float(value.item())

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

                    raw_state = state.raw_state
                    ue_id = int(action.env_action[0])

                    node_emb, graph_emb, req_emb, req_logits, v_pred = self.policy.encode(state)

                    req_dist = Categorical(logits=req_logits)
                    req_idx_t = torch.tensor(action.req_idx, device=self.device)
                    new_lp = req_dist.log_prob(req_idx_t)
                    entropy = req_dist.entropy()

                    req_vec = req_emb[action.req_idx]

                    if action.mode_sampled:
                        mode_dist = Categorical(logits=self.policy.score_mode(req_vec, graph_emb))
                        mode_t = torch.tensor(action.handover_flag, device=self.device)
                        new_lp = new_lp + mode_dist.log_prob(mode_t)
                        entropy = entropy + mode_dist.entropy()

                    if action.node_sampled:
                        node_logits = self.policy.score_nodes(req_vec, node_emb, graph_emb)
                        ru_logits = node_logits[state.ru_mask]
                        du_logits = node_logits[state.du_mask]
                        cu_logits = node_logits[state.cu_mask]

                        ru_t = torch.tensor(action.ru_local, device=self.device)
                        du_t = torch.tensor(action.du_local, device=self.device)
                        cu_t = torch.tensor(action.cu_local, device=self.device)

                        dist_ru = Categorical(logits=ru_logits)
                        new_lp = new_lp + dist_ru.log_prob(ru_t)
                        entropy = entropy + dist_ru.entropy()

                        dist_du = Categorical(logits=du_logits)
                        new_lp = new_lp + dist_du.log_prob(du_t)
                        entropy = entropy + dist_du.entropy()

                        dist_cu = Categorical(logits=cu_logits)
                        new_lp = new_lp + dist_cu.log_prob(cu_t)
                        entropy = entropy + dist_cu.entropy()

                    rb_rem, ru_rem, du_rem, cu_rem = self._effective_remaining(raw_state, ue_id)

                    rb_logits = self.policy.score_rb(req_vec, graph_emb)
                    rb_t = torch.tensor(action.rb_idx, device=self.device)
                    # rb_valid = torch.zeros(self.policy.max_RBs_per_UE, dtype=torch.bool, device=self.device)
                    # rb_max = max(1, min(self.policy.max_RBs_per_UE, rb_rem))
                    # rb_valid[:rb_max] = True
                    
                    UE = self._get_ue_raw(
                        raw_state,
                        ue_id,
                    )

                    slice_max_RBs = int(
                        UE.get(
                            "max_RBs",
                            self.policy.max_RBs_per_UE,
                        )
                    )

                    rb_max = max(
                        1,
                        min(
                            self.policy.max_RBs_per_UE,
                            slice_max_RBs,
                            rb_rem,
                        ),
                    )

                    rb_valid = torch.zeros(
                        self.policy.max_RBs_per_UE,
                        dtype=torch.bool,
                        device=self.device,
                    )

                    rb_valid[:rb_max] = True

                    dist_rb = self._masked_categorical(rb_logits, rb_valid)
                    new_lp = new_lp + dist_rb.log_prob(rb_t)
                    entropy = entropy + dist_rb.entropy()

                    chosen_ru_global = action.ru_global
                    chosen_ru_emb = node_emb[chosen_ru_global]
                    power_logits = self.policy.score_power(req_vec, graph_emb, chosen_ru_emb)

                    power_available = float(ru_rem[chosen_ru_global])
                    power_mask = torch.tensor(
                        [float(p) <= power_available + 1e-12 for p in self.power_levels],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    if power_mask.sum() == 0:
                        power_mask[:] = True
                    dist_pw = self._masked_categorical(power_logits, power_mask)
                    pw_t = torch.tensor(action.power_idx, device=self.device)
                    new_lp = new_lp + dist_pw.log_prob(pw_t)
                    entropy = entropy + dist_pw.entropy()

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
            state = env.get_state()

            ep_reward = ep_accept = ep_thr = ep_lat = ep_ho = ep_pingpong = 0.0
            ep_acc_term = ep_thr_term = ep_lat_term = ep_ho_term = 0.0

            done = False
            
            steps = 0
            num_ues = len(env.UE_manager.UE_requests)
            start_time = time.time()
            while not done:
                #print(f"state[{steps}] = {state}")
                cached_state, cached_action, logp, value = agent.select_action(state)
                #print(f"action[{steps}] = {cached_action}")
                if cached_action is None:
                    break

                next_state, reward, done, info = env.step(cached_action.env_action)
                
                # if isinstance(info, dict) and not info.get("success", False):
                #     print(
                #         f"[REJECT] UE={cached_action.env_action[0]} | "
                #         f"Action={cached_action.env_action} | "
                #         f"Info={info}"
                #     )
                #print(f"info[{steps}] = {info}")

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
                    ep_ho_term += float(info.get("handover_term", 0.0))

                state = next_state
                steps_collected += 1
                steps += 1
                #print(f"state[-{steps}] = {state}")

                if steps_collected >= update_per_steps:
                    agent.update(buf, episode_idx=ep)
                    buf = RolloutBuffer(states=[], actions=[], logprobs=[], rewards=[], values=[], next_values=[], masks=[])
                    steps_collected = 0

            elapsed = time.time() - start_time

            removed_ues_with_info, _ = env.UE_manager.UE_mobility()
            for _, ue_info in removed_ues_with_info:
                alloc = ue_info.get("allocation", {})
                if alloc["RU"] is not None:
                    env.resource_manager.release_resources(alloc["RU"], alloc["DU"], alloc["CU"], alloc["num_RB_alloc"], alloc["power_alloc"], alloc["cpu_DU_req"], alloc["cpu_CU_req"])

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
                f"Reward ={ep_reward:7.2f} | "
                f"Accept ={ep_accept:3.0f}/{num_ues:3.0f} | "
                f"Thr = {ep_thr/1e6:6.2f}({avg_thr:6.2f})Mb | "
                f"Lat = {ep_lat*1e3:6.2f}({avg_lat:6.2f})ms | "
                f"HO ={ep_ho:3.0f}({ho_rate:.2f}) | "
                f"PP ={ep_pingpong:3.0f}({pp_rate:.2f})"
            )

        if len(buf.rewards) > 0:
            agent.update(buf, episode_idx=max_episode)

        return agent
    finally:
        for f in files.values():
            f.close()


def evaluate_agent(env, agent, render=False):
    state = env.get_state()

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
            ep_ho_term += float(info.get("handover_term", 0.0))

        state = next_state
        if render:
            print(info)
    
    removed_ues_with_info, _ = env.UE_manager.UE_mobility()
    for _, ue_info in removed_ues_with_info:
        alloc = ue_info.get("allocation", {})
        if alloc["RU"] is not None:
            env.resource_manager.release_resources(alloc["RU"], alloc["DU"], alloc["CU"], alloc["num_RB_alloc"], alloc["power_alloc"], alloc["cpu_DU_req"], alloc["cpu_CU_req"])

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
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

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
        print(f"💾 Đã lưu checkpoint tại: {path}")
    except Exception as e:
        print(f"❌ Lỗi khi lưu checkpoint: {e}")


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
        print(f"⚠️ Không tìm thấy checkpoint tại: {path}")
        return False

    try:
        ckpt = torch.load(path, map_location=agent.device)
        policy_state = ckpt.get("policy_state", {})

        if partial:
            loaded, skipped = load_partial_policy_state(agent.policy, policy_state, strict=False)
            print(f"✅ Load partial checkpoint từ: {path}")
            print(f"   - Tensor load được: {len(loaded)}")
            print(f"   - Tensor bỏ qua: {len(skipped)}")
        else:
            agent.policy.load_state_dict(policy_state, strict=strict)
            print(f"✅ Đã load checkpoint từ: {path}")

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
        print(f"❌ Lỗi khi load checkpoint từ {path}: {e}")
        return False