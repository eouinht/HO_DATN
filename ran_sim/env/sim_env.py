from __future__ import annotations

from typing import Any, Dict, Optional
import math

from core.config import SimulationConfig
from core.rng import RNGManager

from domain.layout import create_layout
from domain.cells import configure_cells
from domain.ues import initialize_ues
from domain.entities import SimulationResults
from domain.ue_lifecycle import update_ue_lifecycle

from resources.pool import create_resource_pool, reset_resource_pool
from resources.adaptive_allocator import find_feasible_allocation
from resources.allocation import (
    compute_allocation_metrics,
    check_feasible,
    reserve_resources,
    release_resources,
)

from mobility.engine import MobilityEngine
from radio.measurements import MeasurementEngine
from radio.attachment import handle_attachment
from radio.handover import HandoverEngine


class SimRANEnv:
    """
    System-level RAN simulator.

    Pipeline mỗi step:
        1. UE mobility
        2. Signal measurement
        3. Attachment / reconnect / disconnect
        4. Agent action: RU-DU-CU + PRB + power allocation
        5. Handover check
        6. KPI / reward / done
    """

    def __init__(
        self,
        cfg: Optional[SimulationConfig] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg if cfg is not None else SimulationConfig()
        self.seed = self.cfg.seed if seed is None else seed

        self.rng_manager = RNGManager(self.seed)

        self.mobility_engine = MobilityEngine()
        self.measurement_engine = MeasurementEngine()
        self.handover_engine = HandoverEngine()

        self.sites = []
        self.cells = []
        self.ues = []
        self.resource_pool = None

        self.results = SimulationResults()

        self.current_time = 0.0
        self.step_idx = 0

        self.total_reward = 0.0
        self.total_handovers = 0
        self.successful_handovers = 0

    # =====================================================
    # Reset
    # =====================================================

    def reset(self) -> Dict[str, Any]:
        self.rng_manager = RNGManager(self.seed)

        self.current_time = 0.0
        self.step_idx = 0

        self.results = SimulationResults()
        self.total_reward = 0.0
        self.total_handovers = 0
        self.successful_handovers = 0

        self.sites = create_layout(self.cfg, self.rng_manager)
        self.cells = configure_cells(self.sites, self.cfg, self.rng_manager)
        self.ues = initialize_ues(self.cfg, self.rng_manager)

        self.resource_pool = create_resource_pool(self.cfg, self.rng_manager)

        # Initial measurement + attachment
        self.ues = self.measurement_engine.update(
            self.ues,
            self.cells,
            self.cfg,
            self.current_time,
            self.rng_manager,
        )

        self.ues = handle_attachment(
            self.ues,
            self.cells,
            self.cfg,
            self.current_time,
        )

        return self.get_state()

    # =====================================================
    # Step
    # =====================================================

    def step(self, action: Dict[str, Any]):
        """
        action dạng tối thiểu:

        {
            "ue_id": 1,
            "cell_id": 3,
            "du_id": 1,
            "cu_id": 2,
            "num_prbs": 5,
            "tx_power_watts": 2.0
        }
        """

        self.step_idx += 1
        self.current_time = self.step_idx * self.cfg.time_step

        # 1. Mobility
        self.ues = self.mobility_engine.update(
            self.ues,
            self.cfg,
            self.current_time,
            self.rng_manager,
        )

        # 1.5 UE lifecycle: arrivals / departures
        lifecycle_info = {
            "num_arrivals": 0,
            "num_departures": 0,
            "arrival_ids": [],
            "departure_ids": [],
        }

        if self.cfg.enable_dynamic_ues:
            self.ues, lifecycle_info = update_ue_lifecycle(
                ues=self.ues,
                cfg=self.cfg,
                current_time=self.current_time,
                rng_manager=self.rng_manager,
            )
    
        # 2. Measurement
        self.ues = self.measurement_engine.update(
            self.ues,
            self.cells,
            self.cfg,
            self.current_time,
            self.rng_manager,
        )

        # 3. Attachment / reconnect / disconnect
        self.ues = handle_attachment(
            self.ues,
            self.cells,
            self.cfg,
            self.current_time,
        )

        # 4. Apply action
        reward, info = self.apply_action(action)

        # 5. Handover
        ho_events, self.ues = self.handover_engine.check(
            self.ues,
            self.cells,
            self.cfg,
            self.current_time,
            self.rng_manager,
        )

        for event in ho_events:
            self.results.handover_events.append(event)
            self.total_handovers += 1
            if event.ho_success:
                self.successful_handovers += 1

        # 6. Update reward
        self.total_reward += reward

        done = self.is_done()

        state = self.get_state()

        info.update({
            "time": self.current_time,
            "step_idx": self.step_idx,
            "num_arrivals": lifecycle_info["num_arrivals"],
            "num_departures": lifecycle_info["num_departures"],
            "arrival_ids": lifecycle_info["arrival_ids"],
            "departure_ids": lifecycle_info["departure_ids"],
            "num_handover_events": len(ho_events),
            "total_handovers": self.total_handovers,
            "handover_success_rate": self.successful_handovers / max(1, self.total_handovers),
        })

        return state, reward, done, info

    # =====================================================
    # Apply action
    # =====================================================

    def apply_action(self, action: Dict[str, Any]):
        """
        Action dạng:

        {
            "ue_id": 1,
            "handover_flag": 0 or 1,
            "cell_id": 3,
            "du_id": 1,
            "cu_id": 2,
            "num_prbs": 5,
            "tx_power_watts": 2.0
        }
        """

        ue_id = int(action["ue_id"])
        handover_flag = int(action.get("handover_flag", 0))
        cell_id = int(action["cell_id"])
        du_id = int(action["du_id"])
        cu_id = int(action["cu_id"])
        num_prbs = int(action["num_prbs"])
        tx_power_watts = float(action["tx_power_watts"])

        ue = self.get_ue_by_id(ue_id)
        cell = self.get_cell_by_id(cell_id)

        # =====================================================
        # 1. Basic action validation
        # =====================================================
        valid, reason = self.check_action_valid(
            ue_id=ue_id,
            handover_flag=handover_flag,
            cell_id=cell_id,
            du_id=du_id,
            cu_id=cu_id,
            num_prbs=num_prbs,
            tx_power_watts=tx_power_watts,
        )

        if not valid:
            return self.compute_failure_reward(reason), {
                "success": False,
                "reason": reason,
                "ue_id": ue_id,
                "cell_id": cell_id,
            }

        assert ue is not None
        assert cell is not None

        if not ue.connected:
            return -5.0, {
                "success": False,
                "reason": "UE not connected",
                "ue_id": ue_id,
                "cell_id": cell_id,
            }

        # =====================================================
        # 2. Save previous allocation state
        # =====================================================
        prev_alloc = ue.allocation
        prev_prev_alloc = ue.prev_allocation

        old_alloc_exists = prev_alloc.ru_id is not None

        prev_ru = prev_alloc.ru_id
        prev_du = prev_alloc.du_id
        prev_cu = prev_alloc.cu_id

        target_ru = cell.ru_id
        target_du = du_id
        target_cu = cu_id

        is_handover = 0
        is_pingpong = 0

        # =====================================================
        # 3. Release old allocation before checking new one
        # =====================================================
        if old_alloc_exists:
            release_resources(prev_alloc, self.resource_pool)

        # =====================================================
        # 4. Semantic constraints: keep mapping / handover
        # =====================================================

        # CASE A: UE already has allocation and handover_flag = 0
        # => RU/DU/CU must stay the same
        if old_alloc_exists and handover_flag == 0:
            if target_ru != prev_ru or target_du != prev_du or target_cu != prev_cu:
                # restore old resource because action invalid
                reserve_resources(
                    allocation=prev_alloc,
                    resource_pool=self.resource_pool,
                    du_id=prev_du,
                    cu_id=prev_cu,
                )

                return 0.0, {
                    "success": False,
                    "reason": "invalid keep mapping: RU/DU/CU changed",
                    "ue_id": ue_id,
                    "cell_id": cell_id,
                }

            is_handover = 0
            is_pingpong = 0

        # CASE B: UE already has allocation and handover_flag = 1
        # => RU must change
        elif old_alloc_exists and handover_flag == 1:
            if target_ru == prev_ru:
                # restore old resource because action invalid
                reserve_resources(
                    allocation=prev_alloc,
                    resource_pool=self.resource_pool,
                    du_id=prev_du,
                    cu_id=prev_cu,
                )

                return 0.0, {
                    "success": False,
                    "reason": "invalid handover: target RU equals previous RU",
                    "ue_id": ue_id,
                    "cell_id": cell_id,
                }

            is_handover = 1

            if (
                prev_prev_alloc.ru_id is not None
                and target_ru == prev_prev_alloc.ru_id
            ):
                is_pingpong = ue.pingpong_count + 1
            else:
                is_pingpong = 0

        # CASE C: UE has no previous allocation
        # => initial allocation, not real handover
        else:
            is_handover = 0
            is_pingpong = 0

        # =====================================================
        # 5. Compute allocation metrics
        # =====================================================
        allocation = compute_allocation_metrics(
            ue=ue,
            cell=cell,
            cfg=self.cfg,
            rng_manager=self.rng_manager,
            current_time=self.current_time,
            num_prbs=num_prbs,
            tx_power_watts=tx_power_watts,
        )

        feasible, reason = check_feasible(
            allocation=allocation,
            resource_pool=self.resource_pool,
            ue=ue,
            du_id=du_id,
            cu_id=cu_id,
        )

        # =====================================================
        # 6. If infeasible, do not accept allocation
        # =====================================================
        if not feasible:
            # restore old allocation if it existed
            if old_alloc_exists:
                reserve_resources(
                    allocation=prev_alloc,
                    resource_pool=self.resource_pool,
                    du_id=prev_du,
                    cu_id=prev_cu,
                )

            return self.compute_failure_reward(reason), {
                "success": False,
                "reason": reason,
                "ue_id": ue_id,
                "cell_id": cell_id,
                "handover": is_handover,
                "pingpong": is_pingpong,
                "throughput_bps": allocation.throughput_bps,
                "latency_s": allocation.latency_s,
                "num_prbs": allocation.num_prbs,
                "tx_power_watts": allocation.tx_power_watts,
            }

        # =====================================================
        # 7. Reserve new resources
        # =====================================================
        allocation = reserve_resources(
            allocation=allocation,
            resource_pool=self.resource_pool,
            du_id=du_id,
            cu_id=cu_id,
        )

        # =====================================================
        # 8. Update UE state
        # =====================================================
        ue.prev_allocation = prev_alloc
        ue.allocation = allocation

        ue.serving_cell = cell.id
        ue.serving_ru = cell.ru_id
        ue.connected = True

        ue.handover_count += is_handover
        ue.pingpong_count = is_pingpong

        # =====================================================
        # 9. Compute reward
        # =====================================================
        reward = self.compute_success_reward(
            ue=ue,
            is_handover=is_handover,
            is_pingpong=is_pingpong,
        )

        return reward, {
            "success": True,
            "reason": "allocation_success",
            "ue_id": ue_id,
            "cell_id": cell_id,
            "du_id": du_id,
            "cu_id": cu_id,
            "ru_id": cell.ru_id,
            "throughput_bps": allocation.throughput_bps,
            "latency_s": allocation.latency_s,
            "num_prbs": allocation.num_prbs,
            "tx_power_watts": allocation.tx_power_watts,
            "handover": is_handover,
            "pingpong": is_pingpong,
        }

    # =====================================================
    # Reward
    # =====================================================

    def compute_success_reward(self, ue, is_handover: int=0, is_pingpong: int=0) -> float:
        alloc = ue.allocation

        rate_term = alloc.throughput_bps / max(1e-9, ue.min_rate_bps)
        latency_term = ue.max_latency_s / max(1e-9, alloc.latency_s)

        handover_penalty = 0.05 * ue.handover_count
        pingpong_penalty = 0.1 * ue.pingpong_count
        handover_term = 0.25 * is_handover * (1.0 + math.exp(is_pingpong))
        reward = (
            1.0
            + 0.5 * rate_term
            + 0.5 * latency_term
            - handover_term
        )

        return float(reward)

    def compute_failure_reward(self, reason: str) -> float:
        if "QoS" in reason:
            return -5.0
        if "resource" in reason:
            return -3.0
        if "link" in reason:
            return -4.0
        return -2.0

    def get_power_levels(self):
        import numpy as np

        return np.linspace(
            0.1,
            self.cfg.ru_power_capacity_choices_watts[0],
            10,
            dtype=np.float32,
        ).tolist()
    
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
        if self.get_ue_by_id(ue_id) is None:
            return False, "invalid UE id"

        if handover_flag not in [0, 1]:
            return False, "invalid handover flag"

        if self.get_cell_by_id(cell_id) is None:
            return False, "invalid cell id"

        if not (1 <= du_id <= self.cfg.num_dus):
            return False, "invalid DU id"

        if not (1 <= cu_id <= self.cfg.num_cus):
            return False, "invalid CU id"

        if num_prbs < 1 or num_prbs > self.cfg.max_prbs_per_ue:
            return False, "invalid PRB allocation"

        if num_prbs > self.resource_pool.prbs_remaining:
            return False, "insufficient PRB resource"

        valid_power_levels = self.get_power_levels()
        if not any(abs(tx_power_watts - p) < 1e-4 for p in valid_power_levels):
            return False, "invalid power level"

        return True, "valid"

    # =====================================================
    # State
    # =====================================================

    def get_state(self) -> Dict[str, Any]:
        return {
            "time": self.current_time,
            "step_idx": self.step_idx,
            "network": self.get_network_state(),
            "ues": self.get_ue_state(),
            "cells": self.get_cell_state(),
            "resources": self.get_resource_state(),
        }

    def get_network_state(self) -> Dict[str, Any]:
        connected = sum(1 for ue in self.ues if ue.connected)

        return {
            "num_ues": len(self.ues),
            "connected_ues": connected,
            "connection_rate": connected / max(1, len(self.ues)),
            "total_handovers": self.total_handovers,
            "handover_success_rate": self.successful_handovers / max(1, self.total_handovers),
        }

    def get_ue_state(self) -> list[dict]:
        return [
            {
                "id": ue.id,
                "x": ue.x,
                "y": ue.y,
                "slice_type": ue.slice_type,
                "connected": ue.connected,
                "serving_cell": ue.serving_cell,
                "serving_ru": ue.serving_ru,
                "rsrp": ue.rsrp,
                "rsrq": ue.rsrq,
                "sinr": ue.sinr,
                "traffic_demand_bps": ue.traffic_demand_bps,
                "min_rate_bps": ue.min_rate_bps,
                "max_latency_s": ue.max_latency_s,
                "allocation": {
                    "ru_id": ue.allocation.ru_id,
                    "du_id": ue.allocation.du_id,
                    "cu_id": ue.allocation.cu_id,
                    "num_prbs": ue.allocation.num_prbs,
                    "tx_power_watts": ue.allocation.tx_power_watts,
                    "throughput_bps": ue.allocation.throughput_bps,
                    "latency_s": ue.allocation.latency_s,
                },
            }
            for ue in self.ues
        ]

    def get_cell_state(self) -> list[dict]:
        return [
            {
                "id": cell.id,
                "ru_id": cell.ru_id,
                "site_id": cell.site_id,
                "sector_id": cell.sector_id,
                "x": cell.x,
                "y": cell.y,
                "tx_power_dbm": cell.tx_power_dbm,
                "connected_ues": list(cell.connected_ues),
                "current_load_bps": cell.current_load_bps,
                "avg_sinr": cell.avg_sinr,
            }
            for cell in self.cells
        ]

    def get_resource_state(self) -> Dict[str, Any]:
        pool = self.resource_pool

        return {
            "prbs_remaining": pool.prbs_remaining,
            "prbs_total": pool.total_prbs,

            "ru_power_remaining": [
                ru.power_remaining_watts for ru in pool.rus
            ],
            "ru_power_capacity": [
                ru.power_capacity_watts for ru in pool.rus
            ],

            "du_remaining": [
                du.remaining_cycles_per_s for du in pool.dus
            ],
            "du_capacity": [
                du.capacity_cycles_per_s for du in pool.dus
            ],

            "cu_remaining": [
                cu.remaining_cycles_per_s for cu in pool.cus
            ],
            "cu_capacity": [
                cu.capacity_cycles_per_s for cu in pool.cus
            ],
        }

    # =====================================================
    # Done
    # =====================================================

    def is_done(self) -> bool:
        if self.step_idx >= self.cfg.num_steps:
            return True

        if self.resource_pool.prbs_remaining <= 0:
            return True

        return False

    # =====================================================
    # Helpers
    # =====================================================

    def get_ue_by_id(self, ue_id: int):
        for ue in self.ues:
            if ue.id == ue_id:
                return ue
        return None

    def get_cell_by_id(self, cell_id: int):
        for cell in self.cells:
            if cell.id == cell_id:
                return cell
        return None