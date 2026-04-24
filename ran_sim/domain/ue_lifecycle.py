from __future__ import annotations

from typing import List, Tuple
import math

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import UE
from domain.ues import (
    _sample_position,
    _sample_mobility_pattern,
    _sample_slice_type,
    _get_qos_for_slice,
    _get_traffic_profile,
)


def update_ue_lifecycle(
    ues: List[UE],
    cfg: SimulationConfig,
    current_time: float,
    rng_manager: RNGManager,
) -> Tuple[List[UE], dict]:
    """
    Cập nhật vòng đời UE:
    - UE departure
    - UE arrival
    - giữ số UE trong [min_ues, max_ues]

    Return:
        ues
        info
    """

    time_key = int(current_time * 1000)

    info = {
        "num_arrivals": 0,
        "num_departures": 0,
        "arrival_ids": [],
        "departure_ids": [],
    }

    # =====================================================
    # 1. Departure
    # =====================================================
    if len(ues) > cfg.min_ues:
        rng_departure = rng_manager.get_rng_for(8101, time_key)

        if rng_departure.random() < cfg.ue_departure_probability:
            max_departures = min(
                cfg.max_ue_departures_per_step,
                len(ues) - cfg.min_ues,
            )

            num_departures = rng_departure.randint(0, max_departures)

            if num_departures > 0:
                ues, removed_ids = remove_random_ues(
                    ues=ues,
                    num_remove=num_departures,
                    rng=rng_departure,
                )

                info["num_departures"] = len(removed_ids)
                info["departure_ids"] = removed_ids

    # =====================================================
    # 2. Arrival
    # =====================================================
    if len(ues) < cfg.max_ues:
        rng_arrival = rng_manager.get_rng_for(8201, time_key)

        if rng_arrival.random() < cfg.ue_arrival_probability:
            max_arrivals = min(
                cfg.max_ue_arrivals_per_step,
                cfg.max_ues - len(ues),
            )

            num_arrivals = rng_arrival.randint(0, max_arrivals)

            if len(ues) < cfg.min_ues:
                num_arrivals = max(num_arrivals, cfg.min_ues - len(ues))

            if num_arrivals > 0:
                new_ues = create_new_ues(
                    existing_ues=ues,
                    num_new=num_arrivals,
                    cfg=cfg,
                    current_time=current_time,
                    rng_manager=rng_manager,
                )

                ues.extend(new_ues)

                info["num_arrivals"] = len(new_ues)
                info["arrival_ids"] = [ue.id for ue in new_ues]

    return ues, info


def remove_random_ues(
    ues: List[UE],
    num_remove: int,
    rng,
) -> Tuple[List[UE], List[int]]:
    """
    Xóa ngẫu nhiên một số UE khỏi mạng.
    """

    if num_remove <= 0 or not ues:
        return ues, []

    num_remove = min(num_remove, len(ues))

    indices = list(range(len(ues)))
    rng.shuffle(indices)

    remove_indices = set(indices[:num_remove])

    removed_ids = []
    remaining_ues = []

    for idx, ue in enumerate(ues):
        if idx in remove_indices:
            removed_ids.append(ue.id)
        else:
            remaining_ues.append(ue)

    return remaining_ues, removed_ids


def create_new_ues(
    existing_ues: List[UE],
    num_new: int,
    cfg: SimulationConfig,
    current_time: float,
    rng_manager: RNGManager,
) -> List[UE]:
    """
    Tạo UE mới với id tăng dần.
    """

    if num_new <= 0:
        return []

    max_id = max((ue.id for ue in existing_ues), default=0)
    time_key = int(current_time * 1000)

    new_ues: List[UE] = []

    for i in range(num_new):
        ue_id = max_id + i + 1

        rng = rng_manager.get_rng_for(8301, ue_id, time_key)

        x, y = _sample_position(cfg, rng)
        pattern = _sample_mobility_pattern(cfg, rng)
        direction = rng.uniform(0.0, 2.0 * math.pi)

        slice_type = _sample_slice_type(cfg, rng)
        qos = _get_qos_for_slice(cfg, slice_type)
        packet_size_bits, lambda_pps = _get_traffic_profile(cfg, slice_type)

        ue = UE(
            id=ue_id,
            x=x,
            y=y,

            velocity_mps=cfg.ue_speed_mps,
            direction_rad=direction,
            mobility_pattern=pattern,

            slice_type=slice_type,

            serving_cell=None,
            serving_ru=None,

            traffic_demand_bps=0.0,
            packet_size_bits=packet_size_bits,
            lambda_pps=lambda_pps,
            session_active=False,

            min_rate_bps=qos["min_rate"],
            max_latency_s=qos["max_latency"],
            min_sinr_db=qos["min_sinr"],

            connected=False,
            active=True,
        )

        new_ues.append(ue)

    return new_ues