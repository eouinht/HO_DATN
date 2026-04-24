from typing import List, Tuple
import math

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import UE


# ============================================================
# Public API
# ============================================================

def initialize_ues(cfg: SimulationConfig, rng: RNGManager) -> List[UE]:
    """
    Khởi tạo danh sách UE theo scenario.
    """
    ues: List[UE] = []

    for i in range(cfg.num_ues):
        ue_rng = rng.get_rng_for(i + 1)

        x, y = _sample_position(cfg, ue_rng)
        pattern = _sample_mobility_pattern(cfg, ue_rng)
        direction = ue_rng.uniform(0.0, 2 * math.pi)

        slice_type = _sample_slice_type(cfg, ue_rng)
        qos = _get_qos_for_slice(cfg, slice_type)

        packet_size_bits, lambda_pps = _get_traffic_profile(cfg, slice_type)

        ues.append(
            UE(
                id=i + 1,
                x=x,
                y=y,

                velocity_mps=cfg.ue_speed_mps,
                direction_rad=direction,
                mobility_pattern=pattern,

                slice_type=slice_type,

                # radio state (ban đầu chưa connect)
                serving_cell=None,
                serving_ru=None,

                # traffic
                traffic_demand_bps=0.0,
                packet_size_bits=packet_size_bits,
                lambda_pps=lambda_pps,
                session_active=False,

                # QoS
                min_rate_bps=qos["min_rate"],
                max_latency_s=qos["max_latency"],
                min_sinr_db=qos["min_sinr"],

                # trạng thái ban đầu
                connected=False,
                active=True,
            )
        )

    return ues


# ============================================================
# Position sampling
# ============================================================

def _sample_position(cfg: SimulationConfig, rng) -> Tuple[float, float]:
    if cfg.deployment_scenario == "indoor_hotspot":
        return (
            rng.uniform(0.0, cfg.indoor_width_m),
            rng.uniform(0.0, cfg.indoor_height_m),
        )

    # outdoor: phân bố đều trong hình tròn
    r = cfg.max_radius * math.sqrt(rng.random())
    theta = rng.uniform(0.0, 2 * math.pi)

    x = r * math.cos(theta)
    y = r * math.sin(theta)

    return x, y


# ============================================================
# Mobility
# ============================================================

def _sample_mobility_pattern(cfg: SimulationConfig, rng) -> str:
    if cfg.deployment_scenario == "indoor_hotspot":
        return rng.choice(["stationary", "slow_walk", "normal_walk"])

    if cfg.deployment_scenario == "high_speed":
        return "high_speed_train"

    return rng.choice([
        "pedestrian",
        "slow_vehicle",
        "vehicle",
    ])


# ============================================================
# Slice & QoS
# ============================================================

def _sample_slice_type(cfg: SimulationConfig, rng) -> str:
    if rng.random() < cfg.embb_ratio:
        return "eMBB"
    return "uRLLC"


def _get_qos_for_slice(cfg: SimulationConfig, slice_type: str):
    if slice_type == "eMBB":
        return {
            "min_rate": cfg.embb_min_rate_bps,
            "max_latency": cfg.embb_max_latency_s,
            "min_sinr": cfg.embb_min_sinr_db,
        }
    else:
        return {
            "min_rate": cfg.urllc_min_rate_bps,
            "max_latency": cfg.urllc_max_latency_s,
            "min_sinr": cfg.urllc_min_sinr_db,
        }


# ============================================================
# Traffic profile
# ============================================================

def _get_traffic_profile(cfg: SimulationConfig, slice_type: str):
    if slice_type == "eMBB":
        return cfg.embb_packet_size_bits, cfg.embb_lambda_pps
    else:
        return cfg.urllc_packet_size_bits, cfg.urllc_lambda_pps