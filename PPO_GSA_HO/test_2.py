
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

class GraphStateBuilder:
    def __init__(self, node_in, edge_in, request_in, device):
        self.node_in = int(node_in)
        self.edge_in = int(edge_in)
        self.request_in = int(request_in)
        self.device = device

    def _infer_sizes(self, state):
        RAN = state.get("RAN", {})
        num_RUs = len(np.asarray(RAN.get("RU_power_remaining", []), dtype=float))
        num_DUs = len(np.asarray(RAN.get("DU_remaining", []), dtype=float))
        num_CUs = len(np.asarray(RAN.get("CU_remaining", []), dtype=float))
        total_nodes = num_RUs + num_DUs + num_CUs
        return num_RUs, num_DUs, num_CUs, total_nodes

    def build_node_masks(self, state):
        num_RUs, num_DUs, num_CUs, total_nodes = self._infer_sizes(state)
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
        RAN = state.get("RAN", {})
        RU = np.asarray(RAN.get("RU_power_remaining", []), dtype=float)
        DU = np.asarray(RAN.get("DU_remaining", []), dtype=float)
        CU = np.asarray(RAN.get("CU_remaining", []), dtype=float)
        num_RUs, num_DUs, num_CUs, total_nodes = self._infer_sizes(state)

        if total_nodes == 0:
            return torch.zeros((0, self.node_in), dtype=torch.float32, device=self.device)

        ru_scale = max(float(np.max(RU)) if RU.size else 1.0, 1.0)
        du_scale = max(float(np.max(DU)) if DU.size else 1.0, 1.0)
        cu_scale = max(float(np.max(CU)) if CU.size else 1.0, 1.0)

        RU_n = RU / (ru_scale + 1e-9) if RU.size else np.zeros((0,), dtype=np.float32)
        DU_n = DU / (du_scale + 1e-9) if DU.size else np.zeros((0,), dtype=np.float32)
        CU_n = CU / (cu_scale + 1e-9) if CU.size else np.zeros((0,), dtype=np.float32)

        remain = np.concatenate([RU_n, DU_n, CU_n], axis=0).astype(np.float32)

        link_ru_du = np.asarray(RAN.get("link_bw_ru_du_bps", np.zeros((num_RUs, num_DUs))), dtype=float)
        link_du_cu = np.asarray(RAN.get("link_bw_du_cu_bps", np.zeros((num_DUs, num_CUs))), dtype=float)

        adj = np.zeros((total_nodes, total_nodes), dtype=np.float32)
        if link_ru_du.size > 0:
            for i in range(num_RUs):
                for j in range(num_DUs):
                    bw = float(link_ru_du[i, j])
                    adj[i, num_RUs + j] = bw
                    adj[num_RUs + j, i] = bw
        if link_du_cu.size > 0:
            for j in range(num_DUs):
                for k in range(num_CUs):
                    bw = float(link_du_cu[j, k])
                    adj[num_RUs + j, num_RUs + num_DUs + k] = bw
                    adj[num_RUs + num_DUs + k, num_RUs + j] = bw

        bw_scale = max(
            float(np.max(link_ru_du)) if link_ru_du.size else 1.0,
            float(np.max(link_du_cu)) if link_du_cu.size else 1.0,
            1.0
        )
        degree = (adj > 0).sum(axis=1).astype(np.float32) / max(total_nodes - 1, 1)

        remain_t = torch.from_numpy(remain).to(self.device)
        degree_t = torch.from_numpy(degree).to(self.device)
        remain_t = (remain_t - remain_t.mean()) / (remain_t.std() + 1e-6) if remain_t.numel() > 1 else remain_t * 0.0
        degree_t = degree_t.unsqueeze(-1)

        type_oh = self._node_type_onehot(num_RUs, num_DUs, num_CUs)
        node_feats = torch.cat([remain_t.unsqueeze(-1), degree_t, type_oh], dim=-1)  # (N, 5)
        return node_feats

    def build_edge_features(self, state):
        RAN = state.get("RAN", {})
        RU = np.asarray(RAN.get("RU_power_remaining", []), dtype=float)
        DU = np.asarray(RAN.get("DU_remaining", []), dtype=float)
        CU = np.asarray(RAN.get("CU_remaining", []), dtype=float)
        link_ru_du = np.asarray(RAN.get("link_bw_ru_du_bps", []), dtype=float)
        link_du_cu = np.asarray(RAN.get("link_bw_du_cu_bps", []), dtype=float)
        num_RUs, num_DUs, num_CUs, total_nodes = self._infer_sizes(state)

        if total_nodes == 0:
            return torch.zeros((0, 0, 1), dtype=torch.float32, device=self.device)

        bw_scale = max(
            float(np.max(link_ru_du)) if link_ru_du.size else 1.0,
            float(np.max(link_du_cu)) if link_du_cu.size else 1.0,
            1.0
        )

        adj = np.zeros((total_nodes, total_nodes), dtype=np.float32)

        for i in range(num_RUs):
            for j in range(num_DUs):
                bw = float(link_ru_du[i, j]) / (bw_scale + 1e-9)
                adj[i, num_RUs + j] = bw
                adj[num_RUs + j, i] = bw

        for j in range(num_DUs):
            for k in range(num_CUs):
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

        R_min_list, delay_list, pkt_list, cyc_list, lam_list, gain_best_list, gain_mean_list, dist_best_list = [], [], [], [], [], [], [], []
        for _, UE in active_items:
            R_min_list.append(float(UE.get("R_min", 1.0)))
            delay_list.append(float(UE.get("delay", 1e-3)))
            pkt_list.append(float(UE.get("packet_size_bits", 1.0)))
            cyc_list.append(float(UE.get("cycles_per_packet", 1.0)))
            lam_list.append(float(UE.get("lambda_default_pps", 1.0)))

            gains = np.asarray(UE.get("gain", []), dtype=float)
            dists = np.asarray(UE.get("distances_RU_UE", []), dtype=float)
            gain_best_list.append(float(np.max(gains)) if gains.size else 0.0)
            gain_mean_list.append(float(np.mean(gains)) if gains.size else 0.0)
            dist_best_list.append(float(np.min(dists)) if dists.size else 0.0)

        r_scale = max(max(R_min_list), 1.0)
        d_scale = max(max(delay_list), 1e-6)
        p_scale = max(max(pkt_list), 1.0)
        c_scale = max(max(cyc_list), 1.0)
        l_scale = max(max(lam_list), 1.0)
        g_scale = max(max(gain_best_list), 1.0)
        dist_scale = max(max(dist_best_list), 1.0)

        feats = []
        ids = []
        for ue_id, UE in active_items:
            prev = UE.get("allocation", {})
            has_prev = 1.0 if prev.get("RU") is not None else 0.0

            num_RUs, num_DUs, num_CUs, _ = self._infer_sizes(state)
            prev_RU = float(prev.get("RU", -1))
            prev_DU = float(prev.get("DU", -1))
            prev_CU = float(prev.get("CU", -1))
            prev_RB = float(prev.get("num_RB_alloc", 0.0))
            prev_power = float(prev.get("power_alloc", 0.0))

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

                (float(np.max(np.asarray(UE.get("gain", []), dtype=float))) if len(np.asarray(UE.get("gain", []), dtype=float)) else 0.0) / (g_scale + 1e-9),
                (float(np.mean(np.asarray(UE.get("gain", []), dtype=float))) if len(np.asarray(UE.get("gain", []), dtype=float)) else 0.0) / (g_scale + 1e-9),
                (float(np.min(np.asarray(UE.get("distances_RU_UE", []), dtype=float))) if len(np.asarray(UE.get("distances_RU_UE", []), dtype=float)) else 0.0) / (dist_scale + 1e-9),

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
        )