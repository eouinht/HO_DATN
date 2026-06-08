import copy
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from config import *


class UEManager:
    """
    Bộ sinh dữ liệu UE theo mobility step rời rạc.

    UE rời mạng khi:
        - session kết thúc; hoặc
        - quality-based drop xảy ra.

    Lưu ý:
        - Chưa quy đổi một step thành bao nhiêu giây.
        - Drop dùng reference SNR, chưa dùng SINR.
        - Action reject không đồng nghĩa UE bị drop.
    """

    def __init__(self, coordinates_RU, target_num_UEs=50, log_dir="./logs", enable_logging=True):
        self.coordinates_RU = np.asarray(coordinates_RU, dtype=float)
        if self.coordinates_RU.ndim != 2 or self.coordinates_RU.shape[1] != 2:
            raise ValueError("coordinates_RU must have shape [num_RUs, 2]")

        self.num_RUs = int(self.coordinates_RU.shape[0])
        self.radius_in = 10.0
        self.radius_out = 1000.0

        self.SLICE_PRESET = copy.deepcopy(SLICE_PRESET)
        self.slice_names = list(self.SLICE_PRESET.keys())
        if not self.slice_names:
            raise ValueError("SLICE_PRESET must not be empty")

        self.slice_probabilities = np.asarray(
            [SLICE_PROBABILITIES.get(name, 0.0) for name in self.slice_names],
            dtype=float,
        )
        if self.slice_probabilities.sum() <= 0:
            self.slice_probabilities = np.ones(len(self.slice_names), dtype=float)
        self.slice_probabilities /= self.slice_probabilities.sum()

        self.UE_requests = {}
        self.next_ue_id = 0
        self.current_step = 0

        self.target_num_UEs = int(target_num_UEs)
        self.min_num_UEs = max(1, self.target_num_UEs - UE_POPULATION_MARGIN)
        self.max_num_UEs = self.target_num_UEs + UE_POPULATION_MARGIN

        self.enable_logging = bool(enable_logging)
        self.trajectory_log_file = None
        self.trajectory_log_writer = None
        self.population_log_file = None
        self.population_log_writer = None

        if self.enable_logging:
            self._initialize_loggers(log_dir)

    # =====================================================
    # LOGGING
    # =====================================================

    def _initialize_loggers(self, log_dir):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(log_dir).resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        trajectory_path = log_dir / f"ue_trajectory_{timestamp}.csv"
        population_path = log_dir / f"ue_population_{timestamp}.csv"

        trajectory_fields = [
            "step", "ue_id", "event", "remove_reason", "slice_type",
            "x", "y", "step_distance", "direction_rad",
            "arrival_step", "age_steps", "session_duration_steps", "remaining_session_steps",
            "nearest_ru", "nearest_ru_distance", "best_ru",
            "best_reference_snr_db", "drop_counter",
            "potential_ho", "potential_pingpong", "active", "served",
        ]

        population_fields = [
            "step", "ues_before", "ues_after", "arrivals", "session_ends", "quality_drops",
            "avg_step_distance", "p95_step_distance", "avg_age_steps",
            "avg_remaining_session_steps", "avg_best_reference_snr_db",
            "min_best_reference_snr_db", "max_best_reference_snr_db",
            "potential_ho_count", "potential_pingpong_count",
        ]

        self.trajectory_log_file = open(trajectory_path, "w", newline="", encoding="utf-8", buffering=1)
        self.population_log_file = open(population_path, "w", newline="", encoding="utf-8", buffering=1)

        self.trajectory_log_writer = csv.DictWriter(self.trajectory_log_file, fieldnames=trajectory_fields)
        self.population_log_writer = csv.DictWriter(self.population_log_file, fieldnames=population_fields)

        self.trajectory_log_writer.writeheader()
        self.population_log_writer.writeheader()

        self.trajectory_log_file.flush()
        self.population_log_file.flush()

        print(f"[UE MANAGER LOG] Trajectory: {trajectory_path}")
        print(f"[UE MANAGER LOG] Population: {population_path}")

    def close(self):
        if self.trajectory_log_file is not None:
            self.trajectory_log_file.flush()
            self.trajectory_log_file.close()
            self.trajectory_log_file = None

        if self.population_log_file is not None:
            self.population_log_file.flush()
            self.population_log_file.close()
            self.population_log_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # =====================================================
    # EMPTY STATE
    # =====================================================

    @staticmethod
    def empty_status():
        return {
            "active": 1,
            "served": False,
            "reason": "",
        }

    @staticmethod
    def empty_alloc():
        return {
            "RU": None,
            "DU": None,
            "CU": None,
            "num_RB_alloc": 0,
            "power_alloc": 0.0,
            "cpu_DU_req": 0.0,
            "cpu_CU_req": 0.0,
        }

    # =====================================================
    # SPAWN
    # =====================================================

    def _sample_internal_coordinate(self):
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        radius = float(np.random.uniform(self.radius_in, self.radius_out))
        return float(radius * np.cos(angle)), float(radius * np.sin(angle))

    def _sample_boundary_coordinate_and_direction(self):
        angle = float(np.random.uniform(0.0, 2.0 * np.pi))
        radius = float(
            np.random.uniform(
                BOUNDARY_SPAWN_MIN_RATIO * self.radius_out,
                BOUNDARY_SPAWN_MAX_RATIO * self.radius_out,
            )
        )

        x = float(radius * np.cos(angle))
        y = float(radius * np.sin(angle))

        inward_direction = (
            angle
            + np.pi
            + np.random.uniform(-UE_DIRECTION_JITTER_RAD, UE_DIRECTION_JITTER_RAD)
        ) % (2.0 * np.pi)

        return (x, y), float(inward_direction)

    def set_coordinate_UE(self):
        """
        Giữ tên hàm cũ để tương thích.
        """
        return self._sample_internal_coordinate()

    # =====================================================
    # INITIAL STATE
    # =====================================================

    def initialize_mobility_state(self, initial_direction_rad=None):
        if initial_direction_rad is None:
            initial_direction_rad = float(np.random.uniform(0.0, 2.0 * np.pi))

        return {
            "step_distance": float(np.random.uniform(UE_STEP_DISTANCE_MIN, UE_STEP_DISTANCE_MAX)),
            "direction_rad": float(initial_direction_rad),
            "prev_best_ru": None,
            "prev_prev_best_ru": None,
        }

    def initialize_lifecycle_state(self, initial_population=False):
        """
        UE mới:
            age = 0.

        UE ban đầu:
            lifecycle đã chạy qua một pha ngẫu nhiên để tránh
            nhiều UE kết thúc session đồng thời.
        """
        session_duration_steps = int(
            MIN_SESSION_STEPS + np.random.poisson(MEAN_EXTRA_SESSION_STEPS)
        )
        session_duration_steps = int(
            np.clip(session_duration_steps, MIN_SESSION_STEPS, MAX_SESSION_STEPS)
        )

        if initial_population:
            age_steps = int(np.random.randint(0, session_duration_steps))
            remaining_session_steps = max(1, session_duration_steps - age_steps)
            arrival_step = int(self.current_step - age_steps)
        else:
            arrival_step = int(self.current_step)
            remaining_session_steps = int(session_duration_steps)

        return {
            "arrival_step": arrival_step,
            "session_duration_steps": session_duration_steps,
            "remaining_session_steps": remaining_session_steps,
        }

    # =====================================================
    # RADIO MODEL
    # =====================================================

    def calculate_distances(self, coordinate_UE):
        coordinate_UE = np.asarray(coordinate_UE, dtype=float)
        distances = np.linalg.norm(self.coordinates_RU - coordinate_UE, axis=1)
        return distances.astype(float)

    def calculate_gain(self, distances_RU_UE):
        """
        Trả về:
            path_gain:
                slow gain đã chuẩn hóa theo noise, không gồm fast fading.

            gain:
                gain dùng cho throughput, có thêm Rayleigh fading.

        Nếu project đã có calculate_gain() riêng đã kiểm chứng,
        giữ công thức cũ nhưng nên tiếp tục tách path_gain và gain.
        """
        f_c_ghz = float(globals().get("f_c_GHz", 3.5))
        bandwidth_per_rb = float(globals().get("bandwidth_per_RB", 360e3))
        noise_figure_db = float(globals().get("noise_figure_db", 7.0))
        temperature_k = float(globals().get("temperature_K", 290.0))

        noise_factor = 10.0 ** (noise_figure_db / 10.0)
        noise_power_rb = 1.380649e-23 * temperature_k * bandwidth_per_rb * noise_factor

        path_gains = []
        gains = []

        for distance in distances_RU_UE:
            distance = max(float(distance), 1.0)

            path_loss_db = 28.0 + 20.0 * np.log10(f_c_ghz) + 22.0 * np.log10(distance)
            path_gain_linear = 10.0 ** (-path_loss_db / 10.0)
            normalized_path_gain = path_gain_linear / max(noise_power_rb, 1e-30)

            rayleigh_power_gain = float(np.random.exponential(scale=1.0))
            normalized_gain = normalized_path_gain * rayleigh_power_gain

            path_gains.append(normalized_path_gain)
            gains.append(normalized_gain)

        return np.asarray(path_gains, dtype=float), np.asarray(gains, dtype=float)

    def compute_UE_metrics(self, coordinate_UE):
        distances = self.calculate_distances(coordinate_UE)
        path_gain, gain = self.calculate_gain(distances)
        return distances, path_gain, gain

    # =====================================================
    # CREATE / REMOVE UE
    # =====================================================

    def build_UE_request(self, ue_id, initial_population=False):
        slice_name = str(np.random.choice(self.slice_names, p=self.slice_probabilities))
        spawn_from_boundary = bool(np.random.rand() < BOUNDARY_SPAWN_RATIO)

        if spawn_from_boundary:
            coordinate_UE, initial_direction = self._sample_boundary_coordinate_and_direction()
        else:
            coordinate_UE = self._sample_internal_coordinate()
            initial_direction = float(np.random.uniform(0.0, 2.0 * np.pi))

        distances_RU_UE, path_gain, gain = self.compute_UE_metrics(coordinate_UE)

        ue_info = {
            "id": int(ue_id),
            **copy.deepcopy(self.SLICE_PRESET[slice_name]),
            "coordinate": coordinate_UE,
            "distances_RU_UE": distances_RU_UE,
            "path_gain": path_gain,
            "gain": gain,
            "status": self.empty_status(),
            "allocation": self.empty_alloc(),
            "prev_allocation": self.empty_alloc(),
            "pingpong": 0,
            "handover_count": 0,
            "mobility": self.initialize_mobility_state(initial_direction),
            "lifecycle": self.initialize_lifecycle_state(initial_population=initial_population),
            "coverage": {
                "drop_counter": 0,
                "best_reference_snr_db": None,
            },
        }

        ue_info["coverage"]["best_reference_snr_db"] = self.calculate_best_reference_snr_db(ue_info)

        self.UE_requests[int(ue_id)] = ue_info

        self._log_ue_event(
            ue_info=ue_info,
            event="initial" if initial_population else "arrival",
            remove_reason="",
            potential_ho=False,
            potential_pingpong=False,
        )

        return int(ue_id)

    def add_UEs_requests(self, num_new_UEs, initial_population=False):
        new_ids = []

        for _ in range(max(0, int(num_new_UEs))):
            while self.next_ue_id in self.UE_requests:
                self.next_ue_id += 1

            ue_id = int(self.next_ue_id)
            self.next_ue_id += 1

            self.build_UE_request(
                ue_id=ue_id,
                initial_population=initial_population,
            )

            new_ids.append(ue_id)

        return new_ids

    def remove_UE(self, UE_id):
        """
        Chỉ xóa UE khỏi manager.
        Caller bên ngoài vẫn phải release RB, power và CPU.
        """
        return self.UE_requests.pop(int(UE_id))

    def remove_random_UEs(self, num_UEs_to_remove):
        """
        Giữ hàm cũ để tương thích và debug.
        Không dùng trong lifecycle chính thức.
        """
        num_UEs_to_remove = min(max(int(num_UEs_to_remove), 0), len(self.UE_requests))
        if num_UEs_to_remove == 0:
            return []

        selected_ids = np.random.choice(
            list(self.UE_requests.keys()),
            size=num_UEs_to_remove,
            replace=False,
        )

        removed = []

        for ue_id in selected_ids:
            ue_info = self.remove_UE(int(ue_id))
            ue_info["remove_reason"] = "manual_remove"
            removed.append((int(ue_id), ue_info))

        return removed

    # =====================================================
    # MOBILITY
    # =====================================================

    def adjust_coordinates_UE(self, coordinate_UE, mobility_state):
        x, y = coordinate_UE

        old_step_distance = float(mobility_state["step_distance"])
        distance_jitter = float(
            np.random.uniform(-UE_STEP_DISTANCE_JITTER, UE_STEP_DISTANCE_JITTER)
        )

        new_step_distance = float(
            np.clip(
                old_step_distance + distance_jitter,
                UE_STEP_DISTANCE_MIN,
                UE_STEP_DISTANCE_MAX,
            )
        )

        old_direction = float(mobility_state["direction_rad"])
        direction_jitter = float(
            np.random.uniform(-UE_DIRECTION_JITTER_RAD, UE_DIRECTION_JITTER_RAD)
        )

        new_direction = (old_direction + direction_jitter) % (2.0 * np.pi)

        new_x = float(x) + new_step_distance * np.cos(new_direction)
        new_y = float(y) + new_step_distance * np.sin(new_direction)

        mobility_state["step_distance"] = new_step_distance
        mobility_state["direction_rad"] = float(new_direction)

        return float(new_x), float(new_y)

    # =====================================================
    # QUALITY DROP
    # =====================================================

    def calculate_best_reference_snr_db(self, ue_info):
        """
        Dùng slow path gain thay vì gain có fast fading.
        Điều này tránh drop UE chỉ vì một fading sample xấu.
        """
        path_gains = np.asarray(ue_info.get("path_gain", []), dtype=float)
        if path_gains.size == 0:
            return -float("inf")

        reference_snr_linear = REFERENCE_POWER_PER_RB_W * path_gains
        best_reference_snr_linear = float(np.max(reference_snr_linear))

        return float(
            10.0
            * np.log10(
                max(best_reference_snr_linear, 1e-30)
            )
        )

    def update_drop_state(self, ue_info):
        best_snr_db = self.calculate_best_reference_snr_db(ue_info)

        coverage = ue_info.setdefault(
            "coverage",
            {
                "drop_counter": 0,
                "best_reference_snr_db": None,
            },
        )

        coverage["best_reference_snr_db"] = best_snr_db

        age_steps = self.current_step - int(ue_info["lifecycle"]["arrival_step"])

        if age_steps < NEW_UE_DROP_GRACE_STEPS:
            coverage["drop_counter"] = 0
            return False, best_snr_db

        if best_snr_db < SNR_DROP_THRESHOLD_DB:
            coverage["drop_counter"] = int(coverage.get("drop_counter", 0)) + 1
        else:
            coverage["drop_counter"] = 0

        should_drop = coverage["drop_counter"] >= DROP_TTT_STEPS
        return bool(should_drop), float(best_snr_db)

    # =====================================================
    # SESSION LIFECYCLE
    # =====================================================

    def update_lifecycle_state(self, ue_info):
        lifecycle = ue_info["lifecycle"]

        lifecycle["remaining_session_steps"] = max(
            0,
            int(lifecycle["remaining_session_steps"]) - 1,
        )

        return bool(lifecycle["remaining_session_steps"] <= 0)

    # =====================================================
    # POTENTIAL HO / PING-PONG
    # =====================================================

    def _update_potential_handover_state(self, ue_info):
        """
        Potential HO dùng slow path gain thay vì fast fading.
        Nhờ đó HO opportunity phản ánh mobility thay vì nhiễu tức thời.
        """
        path_gains = np.asarray(ue_info["path_gain"], dtype=float)
        best_ru = int(np.argmax(path_gains))

        mobility = ue_info["mobility"]
        prev_best_ru = mobility.get("prev_best_ru")
        prev_prev_best_ru = mobility.get("prev_prev_best_ru")

        potential_ho = bool(
            prev_best_ru is not None
            and best_ru != int(prev_best_ru)
        )

        potential_pingpong = bool(
            prev_prev_best_ru is not None
            and prev_best_ru is not None
            and best_ru == int(prev_prev_best_ru)
            and best_ru != int(prev_best_ru)
        )

        mobility["prev_prev_best_ru"] = prev_best_ru
        mobility["prev_best_ru"] = best_ru

        return best_ru, potential_ho, potential_pingpong

    # =====================================================
    # CONTROLLED ARRIVAL
    # =====================================================

    def rebalance_UE_population(self):
        """
        Sinh arrival có mean reversion.

        - Trong vùng ổn định: arrival theo Poisson.
        - Khi population thấp hơn min: recovery nhanh hơn.
        - Không remove ngẫu nhiên.
        """
        current_num_ues = len(self.UE_requests)

        if current_num_ues >= self.max_num_UEs:
            return []

        population_error = self.target_num_UEs - current_num_ues

        arrival_lambda = max(
            0.0,
            BASE_ARRIVAL_LAMBDA + POPULATION_CORRECTION_GAIN * population_error,
        )

        num_arrivals = int(np.random.poisson(arrival_lambda))
        num_arrivals = min(num_arrivals, MAX_UE_ARRIVALS_PER_STEP)

        if current_num_ues < self.min_num_UEs:
            recovery_needed = self.min_num_UEs - current_num_ues

            num_arrivals = max(
                num_arrivals,
                min(
                    recovery_needed,
                    MAX_RECOVERY_ARRIVALS_PER_STEP,
                ),
            )

        num_arrivals = min(
            num_arrivals,
            self.max_num_UEs - current_num_ues,
        )

        if num_arrivals <= 0:
            return []

        return self.add_UEs_requests(
            num_new_UEs=num_arrivals,
            initial_population=False,
        )

    # =====================================================
    # LOGGING HELPERS
    # =====================================================

    def _log_ue_event(
        self,
        ue_info,
        event,
        remove_reason,
        potential_ho,
        potential_pingpong,
    ):
        if not self.enable_logging:
            return

        distances = np.asarray(ue_info["distances_RU_UE"], dtype=float)
        path_gains = np.asarray(ue_info["path_gain"], dtype=float)

        nearest_ru = int(np.argmin(distances))
        best_ru = int(np.argmax(path_gains))

        lifecycle = ue_info["lifecycle"]
        mobility = ue_info["mobility"]
        coverage = ue_info["coverage"]
        status = ue_info["status"]

        x, y = ue_info["coordinate"]

        self.trajectory_log_writer.writerow(
            {
                "step": int(self.current_step),
                "ue_id": int(ue_info["id"]),
                "event": str(event),
                "remove_reason": str(remove_reason),
                "slice_type": str(ue_info.get("type", "unknown")),
                "x": float(x),
                "y": float(y),
                "step_distance": float(mobility["step_distance"]),
                "direction_rad": float(mobility["direction_rad"]),
                "arrival_step": int(lifecycle["arrival_step"]),
                "age_steps": int(self.current_step - lifecycle["arrival_step"]),
                "session_duration_steps": int(lifecycle["session_duration_steps"]),
                "remaining_session_steps": int(lifecycle["remaining_session_steps"]),
                "nearest_ru": nearest_ru,
                "nearest_ru_distance": float(distances[nearest_ru]),
                "best_ru": best_ru,
                "best_reference_snr_db": float(
                    coverage.get("best_reference_snr_db", np.nan)
                ),
                "drop_counter": int(coverage.get("drop_counter", 0)),
                "potential_ho": int(bool(potential_ho)),
                "potential_pingpong": int(bool(potential_pingpong)),
                "active": int(status.get("active", 0)),
                "served": int(bool(status.get("served", False))),
            }
        )

        self.trajectory_log_file.flush()

    def _log_population_step(
        self,
        ues_before,
        arrivals,
        session_ends,
        quality_drops,
        potential_ho_count,
        potential_pingpong_count,
    ):
        if not self.enable_logging:
            return

        current_ues = list(self.UE_requests.values())

        if not current_ues:
            stats = {
                "avg_step_distance": np.nan,
                "p95_step_distance": np.nan,
                "avg_age_steps": np.nan,
                "avg_remaining_session_steps": np.nan,
                "avg_best_reference_snr_db": np.nan,
                "min_best_reference_snr_db": np.nan,
                "max_best_reference_snr_db": np.nan,
            }
        else:
            step_distances = np.asarray(
                [ue["mobility"]["step_distance"] for ue in current_ues],
                dtype=float,
            )

            age_steps = np.asarray(
                [
                    self.current_step - int(ue["lifecycle"]["arrival_step"])
                    for ue in current_ues
                ],
                dtype=float,
            )

            remaining_steps = np.asarray(
                [
                    ue["lifecycle"]["remaining_session_steps"]
                    for ue in current_ues
                ],
                dtype=float,
            )

            best_snrs = np.asarray(
                [
                    ue["coverage"].get("best_reference_snr_db", np.nan)
                    for ue in current_ues
                ],
                dtype=float,
            )

            finite_snrs = best_snrs[np.isfinite(best_snrs)]

            stats = {
                "avg_step_distance": float(np.mean(step_distances)),
                "p95_step_distance": float(np.percentile(step_distances, 95)),
                "avg_age_steps": float(np.mean(age_steps)),
                "avg_remaining_session_steps": float(np.mean(remaining_steps)),
                "avg_best_reference_snr_db": float(np.mean(finite_snrs)) if finite_snrs.size else np.nan,
                "min_best_reference_snr_db": float(np.min(finite_snrs)) if finite_snrs.size else np.nan,
                "max_best_reference_snr_db": float(np.max(finite_snrs)) if finite_snrs.size else np.nan,
            }

        self.population_log_writer.writerow(
            {
                "step": int(self.current_step),
                "ues_before": int(ues_before),
                "ues_after": int(len(self.UE_requests)),
                "arrivals": int(arrivals),
                "session_ends": int(session_ends),
                "quality_drops": int(quality_drops),
                "potential_ho_count": int(potential_ho_count),
                "potential_pingpong_count": int(potential_pingpong_count),
                **stats,
            }
        )

        self.population_log_file.flush()

    # =====================================================
    # ENV UPDATE HELPER
    # =====================================================

    def update_UE_request(self, UE_id, updates):
        UE_id = int(UE_id)

        if UE_id not in self.UE_requests:
            raise KeyError(f"UE_id={UE_id} does not exist")

        for key, value in updates.items():
            self.UE_requests[UE_id][key] = value

    # =====================================================
    # MAIN STEP
    # =====================================================

    def UE_mobility(self):
        """
        Một mobility step:
            1. Di chuyển toàn bộ UE.
            2. Cập nhật radio metrics.
            3. Đếm potential HO và ping-pong.
            4. Xóa UE quality drop.
            5. Xóa UE session end.
            6. Thêm UE mới có kiểm soát.
            7. Ghi log tổng hợp.
        """
        self.current_step += 1

        ues_before = len(self.UE_requests)

        removed_ues_with_info = []

        num_quality_drops = 0
        num_session_ends = 0

        potential_ho_count = 0
        potential_pingpong_count = 0

        for UE_id in list(self.UE_requests.keys()):
            UE = self.UE_requests[UE_id]

            new_coordinate = self.adjust_coordinates_UE(
                UE["coordinate"],
                UE["mobility"],
            )

            distances_RU_UE, path_gain, gain = self.compute_UE_metrics(
                new_coordinate
            )

            UE["coordinate"] = new_coordinate
            UE["distances_RU_UE"] = distances_RU_UE
            UE["path_gain"] = path_gain
            UE["gain"] = gain

            _, potential_ho, potential_pingpong = (
                self._update_potential_handover_state(UE)
            )

            potential_ho_count += int(potential_ho)
            potential_pingpong_count += int(potential_pingpong)

            should_drop, _ = self.update_drop_state(UE)

            if should_drop:
                self._log_ue_event(
                    ue_info=UE,
                    event="quality_drop",
                    remove_reason="quality_drop",
                    potential_ho=potential_ho,
                    potential_pingpong=potential_pingpong,
                )

                removed_UE = self.remove_UE(UE_id)
                removed_UE["remove_reason"] = "quality_drop"
                removed_ues_with_info.append((int(UE_id), removed_UE))
                num_quality_drops += 1
                continue

            should_end_session = self.update_lifecycle_state(UE)

            if should_end_session:
                self._log_ue_event(
                    ue_info=UE,
                    event="session_end",
                    remove_reason="session_end",
                    potential_ho=potential_ho,
                    potential_pingpong=potential_pingpong,
                )

                removed_UE = self.remove_UE(UE_id)
                removed_UE["remove_reason"] = "session_end"
                removed_ues_with_info.append((int(UE_id), removed_UE))
                num_session_ends += 1
                continue

            UE["status"]["active"] = 1

            self._log_ue_event(
                ue_info=UE,
                event="stay",
                remove_reason="",
                potential_ho=potential_ho,
                potential_pingpong=potential_pingpong,
            )

        id_new_UE = self.rebalance_UE_population()

        self._log_population_step(
            ues_before=ues_before,
            arrivals=len(id_new_UE),
            session_ends=num_session_ends,
            quality_drops=num_quality_drops,
            potential_ho_count=potential_ho_count,
            potential_pingpong_count=potential_pingpong_count,
        )

        print(
            f"[MOBILITY] Step={self.current_step} | "
            f"UEs={len(self.UE_requests)} | "
            f"Added={len(id_new_UE)} | "
            f"SessionEnd={num_session_ends} | "
            f"Dropped={num_quality_drops} | "
            f"PotentialHO={potential_ho_count} | "
            f"PotentialPP={potential_pingpong_count}"
        )

        return removed_ues_with_info, id_new_UE