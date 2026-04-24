import math
from domain.entities import UE


def move_ue(ue: UE, distance_m: float) -> None:
    ue.x += distance_m * math.cos(ue.direction_rad)
    ue.y += distance_m * math.sin(ue.direction_rad)


def handle_stationary_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    # UE đứng yên, thỉnh thoảng dao động rất nhỏ
    if rng.random() < 0.05:
        ue.x += rng.uniform(-1.0, 1.0)
        ue.y += rng.uniform(-1.0, 1.0)


def handle_pedestrian_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    distance = ue.velocity_mps * time_step

    if ue.pause_timer_s > 0:
        ue.pause_timer_s -= time_step
        return

    if rng.random() < 0.10:
        ue.pause_timer_s = rng.uniform(5.0, 15.0)
        return

    if rng.random() < 0.30:
        ue.direction_rad += rng.uniform(-math.pi / 2, math.pi / 2)

    move_ue(ue, distance)


def handle_slow_walk_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    original_velocity = ue.velocity_mps
    ue.velocity_mps = min(ue.velocity_mps, 0.5)
    handle_pedestrian_mobility(ue, time_step, current_time, rng)
    ue.velocity_mps = original_velocity


def handle_normal_walk_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    original_velocity = ue.velocity_mps
    ue.velocity_mps = min(ue.velocity_mps, 1.5)
    handle_pedestrian_mobility(ue, time_step, current_time, rng)
    ue.velocity_mps = original_velocity


def handle_fast_walk_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    original_velocity = ue.velocity_mps
    ue.velocity_mps = min(ue.velocity_mps, 2.5)
    handle_pedestrian_mobility(ue, time_step, current_time, rng)
    ue.velocity_mps = original_velocity


def handle_slow_vehicle_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    original_velocity = ue.velocity_mps
    ue.velocity_mps = max(ue.velocity_mps, 5.0)

    distance = ue.velocity_mps * time_step

    if current_time - ue.last_direction_change_s > rng.uniform(30.0, 60.0):
        ue.direction_rad += rng.uniform(-math.pi / 6, math.pi / 6)
        ue.last_direction_change_s = current_time

    move_ue(ue, distance)

    ue.velocity_mps = original_velocity


def handle_vehicle_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    original_velocity = ue.velocity_mps
    ue.velocity_mps = max(ue.velocity_mps, 10.0)

    distance = ue.velocity_mps * time_step

    if current_time - ue.last_direction_change_s > rng.uniform(25.0, 40.0):
        ue.direction_rad += rng.uniform(-math.pi / 4, math.pi / 4)
        ue.last_direction_change_s = current_time

    move_ue(ue, distance)

    ue.velocity_mps = original_velocity


def handle_fast_vehicle_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    original_velocity = ue.velocity_mps
    ue.velocity_mps = max(ue.velocity_mps, 20.0)

    distance = ue.velocity_mps * time_step

    if current_time - ue.last_direction_change_s > rng.uniform(40.0, 60.0):
        ue.direction_rad += rng.uniform(-math.pi / 8, math.pi / 8)
        ue.last_direction_change_s = current_time

    move_ue(ue, distance)

    ue.velocity_mps = original_velocity


def handle_indoor_pedestrian_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    distance = min(ue.velocity_mps, 1.2) * time_step

    if ue.pause_timer_s > 0:
        ue.pause_timer_s -= time_step
        return

    if rng.random() < 0.15:
        ue.pause_timer_s = rng.uniform(2.0, 10.0)
        return

    if rng.random() < 0.40:
        ue.direction_rad += rng.uniform(-math.pi / 2, math.pi / 2)

    move_ue(ue, distance)


def handle_indoor_mobile_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    distance = min(ue.velocity_mps, 2.0) * time_step

    if rng.random() < 0.20:
        ue.direction_rad += rng.uniform(-math.pi / 2, math.pi / 2)

    move_ue(ue, distance)


def handle_outdoor_vehicle_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    original_velocity = ue.velocity_mps
    ue.velocity_mps = max(ue.velocity_mps, 30.0 / 3.6)

    distance = ue.velocity_mps * time_step

    if current_time - ue.last_direction_change_s > rng.uniform(30.0, 50.0):
        ue.direction_rad += rng.uniform(-math.pi / 6, math.pi / 6)
        ue.last_direction_change_s = current_time

    move_ue(ue, distance)

    ue.velocity_mps = original_velocity


def handle_high_speed_train_mobility(ue: UE, time_step: float, current_time: float, rng) -> None:
    # Nếu chưa gán thông tin train, cho chạy như fast vehicle
    if not ue.active:
        return

    distance = max(ue.velocity_mps, 138.89) * time_step  # 500 km/h ≈ 138.89 m/s

    ue.x += distance

    # dao động nhỏ theo trục y
    if rng.random() < 0.05:
        ue.y += rng.uniform(-0.25, 0.25)

    # wrap-around nếu vượt track
    if ue.track_length_m > 0:
        track_end = ue.train_start_x_m + ue.track_length_m
        if ue.x >= track_end:
            ue.x = ue.train_start_x_m + ue.position_in_train_m


def get_mobility_handler(pattern: str):
    handlers = {
        "stationary": handle_stationary_mobility,
        "pedestrian": handle_pedestrian_mobility,
        "slow_walk": handle_slow_walk_mobility,
        "normal_walk": handle_normal_walk_mobility,
        "fast_walk": handle_fast_walk_mobility,
        "slow_vehicle": handle_slow_vehicle_mobility,
        "vehicle": handle_vehicle_mobility,
        "fast_vehicle": handle_fast_vehicle_mobility,
        "indoor_pedestrian": handle_indoor_pedestrian_mobility,
        "indoor_mobile": handle_indoor_mobile_mobility,
        "outdoor_vehicle": handle_outdoor_vehicle_mobility,
        "high_speed_train": handle_high_speed_train_mobility,
    }

    return handlers.get(pattern, handle_pedestrian_mobility)