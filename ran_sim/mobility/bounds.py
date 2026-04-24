import math
from domain.entities import UE
from core.config import SimulationConfig


# ============================================================
# Public API
# ============================================================

def enforce_bounds(ue: UE, cfg: SimulationConfig) -> None:
    """
    Chọn rule theo scenario
    """
    if cfg.deployment_scenario == "indoor_hotspot":
        enforce_indoor_bounds(ue, cfg)
    elif cfg.deployment_scenario == "high_speed":
        enforce_high_speed_bounds(ue, cfg)
    else:
        enforce_circular_bounds(ue, cfg)


# ============================================================
# Indoor bounds (rectangle)
# ============================================================

def enforce_indoor_bounds(ue: UE, cfg: SimulationConfig) -> None:
    """
    UE bị giữ trong hình chữ nhật indoor
    """
    if ue.x < 0:
        ue.x = 0
        ue.direction_rad = math.pi - ue.direction_rad
    elif ue.x > cfg.indoor_width_m:
        ue.x = cfg.indoor_width_m
        ue.direction_rad = math.pi - ue.direction_rad

    if ue.y < 0:
        ue.y = 0
        ue.direction_rad = -ue.direction_rad
    elif ue.y > cfg.indoor_height_m:
        ue.y = cfg.indoor_height_m
        ue.direction_rad = -ue.direction_rad


# ============================================================
# Circular bounds (outdoor macro)
# ============================================================

def enforce_circular_bounds(ue: UE, cfg: SimulationConfig) -> None:
    """
    UE bị giữ trong vòng tròn bán kính max_radius
    """
    dist = math.hypot(ue.x, ue.y)

    if dist > cfg.max_radius:
        # scale lại về biên
        scale = cfg.max_radius / dist
        ue.x *= scale
        ue.y *= scale

        # phản xạ hướng (bounce)
        angle = math.atan2(ue.y, ue.x)
        ue.direction_rad = angle + math.pi


# ============================================================
# High-speed bounds (linear track)
# ============================================================

def enforce_high_speed_bounds(ue: UE, cfg: SimulationConfig) -> None:
    """
    UE chạy trên trục x (đường tàu)
    """
    # giữ trong dải y nhỏ
    if ue.y < 80:
        ue.y = 80
    elif ue.y > 120:
        ue.y = 120

    # wrap-around theo track
    if ue.track_length_m > 0:
        track_end = ue.train_start_x_m + ue.track_length_m
        if ue.x >= track_end:
            ue.x = ue.train_start_x_m + ue.position_in_train_m
        elif ue.x < ue.train_start_x_m:
            ue.x = track_end - ue.position_in_train_m