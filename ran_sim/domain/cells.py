from typing import List
import math

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import Site, Cell


def configure_cells(
    sites: List[Site],
    cfg: SimulationConfig,
    rng: RNGManager,
) -> List[Cell]:
    """
    Tạo danh sách Cell từ Site.
    - Mỗi Site ↔ 1 RU (ru_id = site.id)
    - Mỗi RU có num_sectors sector (trừ indoor → 1 sector)
    """
    cells: List[Cell] = []
    cell_id = 1

    for site in sites:
        num_sectors = 1 if cfg.deployment_scenario == "indoor_hotspot" else cfg.num_sectors

        for s in range(num_sectors):
            azimuth = _sector_azimuth_deg(s, num_sectors)

            # jitter rất nhỏ vị trí cell để tránh trùng điểm khi vẽ/compute
            jx = rng.get_rng_for(site.id, s, 1).uniform(-0.5, 0.5)
            jy = rng.get_rng_for(site.id, s, 2).uniform(-0.5, 0.5)

            cells.append(
                Cell(
                    id=cell_id,
                    ru_id=site.id,          # 1 site ↔ 1 RU
                    site_id=site.id,
                    sector_id=s + 1,

                    x=site.x + jx,
                    y=site.y + jy,
                    azimuth_deg=azimuth,

                    frequency_hz=cfg.carrier_frequency,
                    antenna_height_m=cfg.antenna_height,
                    cell_radius_m=cfg.cell_radius,

                    tx_power_dbm=cfg.initial_tx_power_dbm,
                    min_tx_power_dbm=cfg.min_tx_power_dbm,
                    max_tx_power_dbm=cfg.max_tx_power_dbm,

                    a3_offset_db=cfg.default_a3_offset,
                    ttt_ms=cfg.default_ttt_ms,
                )
            )
            cell_id += 1

    return cells


def _sector_azimuth_deg(sector_idx: int, num_sectors: int) -> float:
    """
    Góc azimuth cho sector:
    - 1 sector  → 0°
    - 3 sector  → 0°, 120°, 240°
    - n sector  → chia đều 360°
    """
    if num_sectors <= 1:
        return 0.0
    return (360.0 / num_sectors) * sector_idx