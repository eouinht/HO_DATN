from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from core.config import SimulationConfig
from env.sim_env import SimRANEnv


class RANGymEnv(gym.Env):
    """
    Gymnasium wrapper cho SimRANEnv.

    Action dạng MultiDiscrete:
        [
            ue_idx,
            cell_idx,
            du_idx,
            cu_idx,
            num_prbs,
            power_level_idx
        ]
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: SimulationConfig | None = None):
        super().__init__()

        self.cfg = cfg if cfg is not None else SimulationConfig()
        self.sim = SimRANEnv(self.cfg)

        self.power_levels_watt = np.linspace(
            0.1,
            self.cfg.ru_power_capacity_choices_watts[0],
            10,
            dtype=np.float32,
        )

        self.action_space = spaces.MultiDiscrete([
        self.cfg.max_ues,                         # ue_idx
        2,                                        # handover_flag: 0/1
        self.cfg.num_rus * self.cfg.num_sectors,  # cell_idx
        self.cfg.num_dus,                         # du_idx
        self.cfg.num_cus,                         # cu_idx
        self.cfg.max_prbs_per_ue + 1,             # num_prbs
        len(self.power_levels_watt),              # power level
        ])

        obs_dim = self._get_obs_dim()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.sim = SimRANEnv(self.cfg, seed=seed)
        else:
            self.sim = SimRANEnv(self.cfg)

        state = self.sim.reset()
        obs = self._state_to_obs(state)

        info = {
            "time": state["time"],
            "step_idx": state["step_idx"],
        }

        return obs, info

    def step(self, action):
        decoded_action = self._decode_action(action)

        state, reward, done, info = self.sim.step(decoded_action)

        obs = self._state_to_obs(state)

        terminated = done
        truncated = False

        return obs, reward, terminated, truncated, info

    def _decode_action(self, action):
        ue_idx = int(action[0])
        handover_flag = int(action[1])
        cell_idx = int(action[2])
        du_idx = int(action[3])
        cu_idx = int(action[4])
        num_prbs = int(action[5])
        power_idx = int(action[6])

        active_ues = self.sim.ues
        if len(active_ues) == 0:
            ue_id = 1
        else:
            ue_idx = ue_idx % len(active_ues)
            ue_id = active_ues[ue_idx].id
            
        num_prbs = max(1, num_prbs)

        return {
            "ue_id": ue_id + 1,
            "handover_flag": handover_flag,
            "cell_id": cell_idx + 1,
            "du_id": du_idx + 1,
            "cu_id": cu_idx + 1,
            "num_prbs": num_prbs,
            "tx_power_watts": float(self.power_levels_watt[power_idx]),
        }

    def _state_to_obs(self, state):
        obs = []

        net = state["network"]
        res = state["resources"]

        # =========================
        # Network-level features
        # =========================
        obs.extend([
            net["connected_ues"] / max(1, net["num_ues"]),
            net["connection_rate"],
            net["total_handovers"] / max(1, self.cfg.num_ues),
            net["handover_success_rate"],
            res["prbs_remaining"] / max(1, res["prbs_total"]),
        ])

        # =========================
        # RU power remaining
        # =========================
        for rem, cap in zip(
            res["ru_power_remaining"],
            res["ru_power_capacity"],
        ):
            obs.append(rem / max(1e-9, cap))

        # =========================
        # DU compute remaining
        # =========================
        for rem, cap in zip(
            res["du_remaining"],
            res["du_capacity"],
        ):
            obs.append(rem / max(1e-9, cap))

        # =========================
        # CU compute remaining
        # =========================
        for rem, cap in zip(
            res["cu_remaining"],
            res["cu_capacity"],
        ):
            obs.append(rem / max(1e-9, cap))

        # =========================
        # Per-UE features
        # =========================
        ues = state["ues"][: self.cfg.max_ues]
        for ue in ues:
            obs.extend([
                ue["x"] / max(1e-9, self.cfg.max_radius),
                ue["y"] / max(1e-9, self.cfg.max_radius),
                1.0 if ue["connected"] else 0.0,
                0.0 if ue["serving_cell"] is None else ue["serving_cell"] / max(1, self.cfg.num_rus * self.cfg.num_sectors),
                0.0 if np.isnan(ue["rsrp"]) else ue["rsrp"] / 100.0,
                0.0 if np.isnan(ue["rsrq"]) else ue["rsrq"] / 20.0,
                0.0 if np.isnan(ue["sinr"]) else ue["sinr"] / 30.0,
                ue["traffic_demand_bps"] / max(1e-9, ue["min_rate_bps"]),
            ])
        num_padding = self.cfg.max_ues - len(ues)
        for _ in range(num_padding):
            obs.extend([0.0] * 8)
        return np.array(obs, dtype=np.float32)
    def check_action_valid(
        self,
        ue_id: int,
        handover_flag: int,
        cell_id: int,
        du_id: int,
        cu_id: int,
        num_prbs: int,
        tx_power_watts: float,
    ):
        ue = self.get_ue_by_id(ue_id)
        if ue is None:
            return False, "invalid UE id"

        cell = self.get_cell_by_id(cell_id)
        if cell is None:
            return False, "invalid cell id"

        if handover_flag not in [0, 1]:
            return False, "invalid handover flag"

        if not (1 <= du_id <= self.cfg.num_dus):
            return False, "invalid DU id"

        if not (1 <= cu_id <= self.cfg.num_cus):
            return False, "invalid CU id"

        if num_prbs < 1 or num_prbs > self.cfg.max_prbs_per_ue:
            return False, "invalid PRB allocation"

        if num_prbs > self.resource_pool.prbs_remaining:
            return False, "insufficient PRB resource"

        valid_power_levels = self.get_power_levels()

        if not any(abs(tx_power_watts - p) < 1e-6 for p in valid_power_levels):
            return False, "invalid power level"

        return True, "valid"


    def get_power_levels(self):
        max_power = self.cfg.ru_power_capacity_choices_watts[0]
        num_levels = 10

        return [
            max_power * i / num_levels
            for i in range(1, num_levels + 1)
        ]
    def _get_obs_dim(self):
        network_dim = 5
        ru_dim = self.cfg.num_rus
        du_dim = self.cfg.num_dus
        cu_dim = self.cfg.num_cus
        ue_dim = self.cfg.max_ues * 8

        return network_dim + ru_dim + du_dim + cu_dim + ue_dim