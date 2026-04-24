# traffic/generator.py

from typing import List
import math

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import UE


class TrafficGenerator:
    """
    Sinh traffic theo Poisson process cho từng UE.

    - arrival: Poisson(lambda_pps)
    - demand_bps = packets * packet_size_bits / time_step
    - có peak hour multiplier
    """

    def update(
        self,
        ues: List[UE],
        cfg: SimulationConfig,
        current_time: float,
        rng_manager: RNGManager,
    ) -> List[UE]:

        time_key = int(current_time * 1000)

        for ue in ues:
            rng = rng_manager.get_rng_for(
                ue.id,
                time_key,
                9001,
            )

            # lambda theo slice
            lam = ue.lambda_pps * cfg.peak_hour_multiplier

            # Poisson sampling (Knuth algorithm nếu không dùng numpy)
            packets = _poisson_sample(lam * cfg.time_step, rng)

            # update demand
            bits = packets * ue.packet_size_bits
            ue.traffic_demand_bps = bits / max(cfg.time_step, 1e-9)

            # session active flag
            ue.session_active = packets > 0

        return ues


# ============================================================
# Poisson sampler (không dùng numpy để giữ consistency RNG)
# ============================================================

def _poisson_sample(lam: float, rng) -> int:
    """
    Knuth Poisson sampling:
    lam: expected arrivals trong interval
    """

    if lam <= 0:
        return 0

    L = math.exp(-lam)
    k = 0
    p = 1.0

    while p > L:
        k += 1
        p *= rng.random()

    return k - 1