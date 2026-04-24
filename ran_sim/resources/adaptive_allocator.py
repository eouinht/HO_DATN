from typing import Optional, Tuple

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import UE, Cell, ResourcePool, Allocation
from resources.allocation import (
    compute_allocation_metrics,
    check_feasible,
)


def find_feasible_allocation(
    ue: UE,
    cell: Cell,
    resource_pool: ResourcePool,
    cfg: SimulationConfig,
    rng_manager: RNGManager,
    current_time: float,
    preferred_du_id: int,
    preferred_cu_id: int,
) -> Tuple[Optional[Allocation], Optional[int], Optional[int], str]:
    """
    Tự động tìm allocation đủ tài nguyên cho UE.

    Logic:
    1. Ưu tiên DU/CU agent chọn
    2. Thử tăng PRB từ thấp đến max
    3. Thử tăng power từ thấp đến cao
    4. Nếu DU/CU agent chọn không đủ, thử DU/CU khác
    """

    power_levels = _build_power_levels(cfg)

    du_candidates = _ordered_candidates(
        preferred=preferred_du_id,
        total=cfg.num_dus,
    )

    cu_candidates = _ordered_candidates(
        preferred=preferred_cu_id,
        total=cfg.num_cus,
    )

    best_fail_reason = "no feasible allocation"

    for du_id in du_candidates:
        for cu_id in cu_candidates:
            for num_prbs in range(1, cfg.max_prbs_per_ue + 1):
                for tx_power_watts in power_levels:
                    allocation = compute_allocation_metrics(
                        ue=ue,
                        cell=cell,
                        cfg=cfg,
                        rng_manager=rng_manager,
                        current_time=current_time,
                        num_prbs=num_prbs,
                        tx_power_watts=tx_power_watts,
                    )

                    feasible, reason = check_feasible(
                        allocation=allocation,
                        resource_pool=resource_pool,
                        ue=ue,
                        du_id=du_id,
                        cu_id=cu_id,
                    )

                    if feasible:
                        return allocation, du_id, cu_id, "feasible"

                    best_fail_reason = reason

    return None, None, None, best_fail_reason


def _build_power_levels(cfg: SimulationConfig):
    max_power = cfg.ru_power_capacity_choices_watts[0]
    levels = []

    for i in range(1, 11):
        levels.append(max_power * i / 10.0)

    return levels


def _ordered_candidates(preferred: int, total: int):
    candidates = list(range(1, total + 1))

    if preferred in candidates:
        candidates.remove(preferred)
        return [preferred] + candidates

    return candidates