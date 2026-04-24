from typing import List

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import UE

from mobility.patterns import get_mobility_handler
from mobility.bounds import enforce_bounds


class MobilityEngine:
    """
    Điều phối mobility:
    - chọn handler theo pattern
    - cập nhật vị trí UE
    - enforce boundary
    """

    def update(
        self,
        ues: List[UE],
        cfg: SimulationConfig,
        current_time: float,
        rng_manager: RNGManager,
    ) -> List[UE]:

        for ue in ues:
            ue.step_counter += 1

            # RNG riêng cho từng UE tại time này
            rng = rng_manager.get_rng_for(ue.id, int(current_time * 1000))

            # chọn handler
            handler = get_mobility_handler(ue.mobility_pattern)

            # cập nhật vị trí
            handler(
                ue=ue,
                time_step=cfg.time_step,
                current_time=current_time,
                rng=rng,
            )

            # enforce boundary
            enforce_bounds(ue, cfg)

        return ues