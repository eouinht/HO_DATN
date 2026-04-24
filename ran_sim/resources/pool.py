from typing import Dict, Tuple, List
import random

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import (
    ResourcePool,
    RUNode,
    DUNode,
    CUNode,
    TransportLink,
)


# ============================================================
# Public API
# ============================================================

def create_resource_pool(
    cfg: SimulationConfig,
    rng_manager: RNGManager,
) -> ResourcePool:
    """
    Tạo ResourcePool:
    - PRB global pool
    - RU power vector (num_rus)
    - DU compute vector (num_dus)
    - CU compute vector (num_cus)
    - RU-DU bandwidth matrix (num_rus x num_dus)
    - DU-CU bandwidth matrix (num_dus x num_cus)
    """

    prbs_total = int(cfg.total_prbs)

    rus = _create_rus(cfg, rng_manager)
    dus = _create_dus(cfg, rng_manager)
    cus = _create_cus(cfg, rng_manager)

    ru_du_links = _create_ru_du_links(cfg, rng_manager)
    du_cu_links = _create_du_cu_links(cfg, rng_manager)

    return ResourcePool(
        total_prbs=prbs_total,
        prbs_remaining=prbs_total,

        rus=rus,
        dus=dus,
        cus=cus,

        ru_du_links=ru_du_links,
        du_cu_links=du_cu_links,
    )


def reset_resource_pool(pool: ResourcePool) -> None:
    """
    Reset toàn bộ remaining resource về capacity ban đầu.
    """

    pool.prbs_remaining = pool.total_prbs

    for ru in pool.rus:
        ru.power_remaining_watts = ru.power_capacity_watts

    for du in pool.dus:
        du.remaining_cycles_per_s = du.capacity_cycles_per_s

    for cu in pool.cus:
        cu.remaining_cycles_per_s = cu.capacity_cycles_per_s

    for link in pool.ru_du_links.values():
        link.remaining_bandwidth_bps = link.bandwidth_bps

    for link in pool.du_cu_links.values():
        link.remaining_bandwidth_bps = link.bandwidth_bps


# ============================================================
# RU / DU / CU creation
# ============================================================

def _choice(rng: random.Random, choices: tuple) -> float:
    return float(choices[int(rng.random() * len(choices))])


def _create_rus(
    cfg: SimulationConfig,
    rng_manager: RNGManager,
) -> List[RUNode]:

    rus: List[RUNode] = []

    for i in range(cfg.num_rus):
        rng = rng_manager.get_rng_for(1001, i)

        cap = _choice(rng, cfg.ru_power_capacity_choices_watts)

        rus.append(
            RUNode(
                id=i + 1,
                site_id=i + 1,
                x=0.0,  # không cần thiết ở đây
                y=0.0,
                power_capacity_watts=cap,
                power_remaining_watts=cap,
            )
        )

    return rus


def _create_dus(
    cfg: SimulationConfig,
    rng_manager: RNGManager,
) -> List[DUNode]:

    dus: List[DUNode] = []

    for i in range(cfg.num_dus):
        rng = rng_manager.get_rng_for(2001, i)

        cap = _choice(rng, cfg.du_capacity_choices_cycles_per_s)

        dus.append(
            DUNode(
                id=i + 1,
                capacity_cycles_per_s=cap,
                remaining_cycles_per_s=cap,
            )
        )

    return dus


def _create_cus(
    cfg: SimulationConfig,
    rng_manager: RNGManager,
) -> List[CUNode]:

    cus: List[CUNode] = []

    for i in range(cfg.num_cus):
        rng = rng_manager.get_rng_for(3001, i)

        cap = _choice(rng, cfg.cu_capacity_choices_cycles_per_s)

        cus.append(
            CUNode(
                id=i + 1,
                capacity_cycles_per_s=cap,
                remaining_cycles_per_s=cap,
            )
        )

    return cus


# ============================================================
# Xhaul links
# ============================================================

def _create_ru_du_links(
    cfg: SimulationConfig,
    rng_manager: RNGManager,
) -> Dict[Tuple[int, int], TransportLink]:

    links: Dict[Tuple[int, int], TransportLink] = {}

    for ru_id in range(1, cfg.num_rus + 1):
        for du_id in range(1, cfg.num_dus + 1):

            rng = rng_manager.get_rng_for(4001, ru_id, du_id)

            bw = _choice(rng, cfg.ru_du_bandwidth_choices_bps)

            links[(ru_id, du_id)] = TransportLink(
                src_id=ru_id,
                dst_id=du_id,
                bandwidth_bps=bw,
                remaining_bandwidth_bps=bw,
                delay_s=cfg.ru_du_delay_s,
                link_type="RU-DU",
            )

    return links


def _create_du_cu_links(
    cfg: SimulationConfig,
    rng_manager: RNGManager,
) -> Dict[Tuple[int, int], TransportLink]:

    links: Dict[Tuple[int, int], TransportLink] = {}

    for du_id in range(1, cfg.num_dus + 1):
        for cu_id in range(1, cfg.num_cus + 1):

            rng = rng_manager.get_rng_for(5001, du_id, cu_id)

            bw = _choice(rng, cfg.du_cu_bandwidth_choices_bps)

            links[(du_id, cu_id)] = TransportLink(
                src_id=du_id,
                dst_id=cu_id,
                bandwidth_bps=bw,
                remaining_bandwidth_bps=bw,
                delay_s=cfg.du_cu_delay_s,
                link_type="DU-CU",
            )

    return links