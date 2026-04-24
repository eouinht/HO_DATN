from typing import List, Optional

from core.config import SimulationConfig
from domain.entities import UE, Cell, NeighborMeasurement


# ============================================================
# Public API
# ============================================================

def handle_attachment(
    ues: List[UE],
    cells: List[Cell],
    cfg: SimulationConfig,
    current_time: float,
) -> List[UE]:
    """
    - UE chưa connect → attach
    - UE yếu tín hiệu → detach
    - UE mất kết nối → reconnect
    """

    for ue in ues:
        if ue.serving_cell is None:
            _try_attach(ue, cfg)
        else:
            _check_disconnect(ue, cfg)

            if not ue.connected:
                _try_reconnect(ue, cfg)

    return ues


# ============================================================
# Attach
# ============================================================

def _try_attach(ue: UE, cfg: SimulationConfig) -> None:
    best = _find_best_cell(ue, cfg.rsrp_serving_threshold)

    if best is None:
        return

    ue.serving_cell = best.cell_id
    ue.serving_ru = best.ru_id

    ue.rsrp = best.rsrp
    ue.rsrq = best.rsrq
    ue.sinr = best.sinr

    ue.connected = True
    ue.connection_timer_s = 0.0


# ============================================================
# Reconnect
# ============================================================

def _try_reconnect(ue: UE, cfg: SimulationConfig) -> None:
    best = _find_best_cell(
        ue,
        cfg.rsrp_serving_threshold + cfg.hysteresis_margin,
    )

    if best is None:
        ue.disconnection_timer_s += 1.0
        return

    ue.serving_cell = best.cell_id
    ue.serving_ru = best.ru_id

    ue.rsrp = best.rsrp
    ue.rsrq = best.rsrq
    ue.sinr = best.sinr

    ue.connected = True
    ue.disconnection_timer_s = 0.0


# ============================================================
# Disconnect
# ============================================================

def _check_disconnect(ue: UE, cfg: SimulationConfig) -> None:
    if ue.rsrp != ue.rsrp:  # NaN check
        _disconnect(ue)
        return

    if ue.rsrp < cfg.rsrp_serving_threshold - cfg.hysteresis_margin:
        ue.disconnection_timer_s += 1.0

        if ue.disconnection_timer_s >= cfg.disconnection_timeout:
            _disconnect(ue)
    else:
        ue.disconnection_timer_s = 0.0


def _disconnect(ue: UE) -> None:
    ue.serving_cell = None
    ue.serving_ru = None
    ue.connected = False

    ue.rsrp = float("nan")
    ue.rsrq = float("nan")
    ue.sinr = float("nan")

    ue.drop_count += 1


# ============================================================
# Helper
# ============================================================

def _find_best_cell(
    ue: UE,
    threshold: float,
) -> Optional[NeighborMeasurement]:

    candidates = [
        m for m in ue.neighbor_measurements
        if m.rsrp >= threshold
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda m: m.rsrp)