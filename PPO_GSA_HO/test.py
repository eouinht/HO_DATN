from pathlib import Path
import csv
import numpy as np

from env.ue import UEManager

# =======================================================
# TEST CONFIGURATION
# =======================================================
NUM_INITIAL_UES = 50
NUM_TEST_STEPS = 1000
LOG_DIR = "./logs/ue_manager_test"

EXPECTED_MIN_UES = 40
EXPECTED_MAX_UES = 60


def build_test_ru_coordinates():
    """
    Tạo topology 7 RU:
        - 1 RU ở trung tâm
        - 6 RU bao quanh

    Khoảng cách giữa các RU đủ gần để tạo vùng giao nhau,
    từ đó có cơ hội phát sinh handover.
    """
    radius = 500.0

    coordinates = [
        [0.0, 0.0],
    ]

    for index in range(6):
        angle = 2.0 * np.pi * index / 6.0

        coordinates.append(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
            ]
        )

    return np.asarray(
        coordinates,
        dtype=float,
    )


def release_dummy_resources(
    removed_ues_with_info,
):
    """
    Test UEManager độc lập nên chưa có ResourceManager.

    Hàm này chỉ thống kê UE rời mạng.
    Khi tích hợp lại vào HandOverEnv, caller vẫn cần
    release RB, power và CPU thật.
    """
    reasons = {
        "quality_drop": 0,
        "session_end": 0,
        "manual_remove": 0,
        "unknown": 0,
    }

    for _, ue_info in removed_ues_with_info:
        reason = ue_info.get(
            "remove_reason",
            "unknown",
        )

        if reason not in reasons:
            reason = "unknown"

        reasons[reason] += 1

    return reasons


def validate_ue_fields(
    manager,
):
    """
    Kiểm tra các field bắt buộc của từng UE.
    """
    required_top_level_fields = [
        "id",
        "coordinate",
        "distances_RU_UE",
        "gain",
        "status",
        "allocation",
        "prev_allocation",
        "mobility",
        "lifecycle",
        "coverage",
    ]

    required_mobility_fields = [
        "step_distance",
        "direction_rad",
        "prev_best_ru",
        "prev_prev_best_ru",
    ]

    required_lifecycle_fields = [
        "arrival_step",
        "session_duration_steps",
        "remaining_session_steps",
    ]

    required_coverage_fields = [
        "drop_counter",
        "best_reference_snr_db",
    ]

    for ue_id, ue in manager.UE_requests.items():
        for field in required_top_level_fields:
            assert field in ue, (
                f"UE={ue_id} missing top-level field: {field}"
            )

        for field in required_mobility_fields:
            assert field in ue["mobility"], (
                f"UE={ue_id} missing mobility field: {field}"
            )

        for field in required_lifecycle_fields:
            assert field in ue["lifecycle"], (
                f"UE={ue_id} missing lifecycle field: {field}"
            )

        for field in required_coverage_fields:
            assert field in ue["coverage"], (
                f"UE={ue_id} missing coverage field: {field}"
            )

        assert len(
            ue["distances_RU_UE"]
        ) == manager.num_RUs, (
            f"UE={ue_id}: invalid distances_RU_UE length"
        )

        assert len(
            ue["gain"]
        ) == manager.num_RUs, (
            f"UE={ue_id}: invalid gain length"
        )

        assert (
            ue["lifecycle"]["remaining_session_steps"]
            >=
            0
        ), (
            f"UE={ue_id}: remaining_session_steps < 0"
        )

        assert (
            ue["mobility"]["step_distance"]
            >=
            0.0
        ), (
            f"UE={ue_id}: negative step_distance"
        )


def validate_population(
    num_ues,
    step,
):
    """
    Dùng khoảng guard rộng hơn khoảng target dự kiến.

    Không fail ngay khi population lệch nhẹ,
    nhưng fail nếu generator trôi mất kiểm soát.
    """
    assert (
        EXPECTED_MIN_UES
        <=
        num_ues
        <=
        EXPECTED_MAX_UES
    ), (
        f"Step={step}: population out of guard range: "
        f"{num_ues} not in "
        f"[{EXPECTED_MIN_UES}, {EXPECTED_MAX_UES}]"
    )


def find_latest_log(
    log_dir,
    pattern,
):
    files = sorted(
        Path(
            log_dir
        ).glob(
            pattern
        ),
        key=lambda path: path.stat().st_mtime,
    )

    if not files:
        raise FileNotFoundError(
            f"No log file matched: {pattern}"
        )

    return files[-1]


def read_population_log(
    population_log_path,
):
    """
    Đọc CSV mà không cần cài pandas.
    """
    rows = []

    with open(
        population_log_path,
        mode="r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            rows.append(
                row
            )

    return rows


def analyze_population_log(
    population_log_path,
):
    rows = read_population_log(
        population_log_path
    )

    if not rows:
        raise RuntimeError(
            "Population log is empty"
        )

    ues_after = np.asarray(
        [
            int(
                row["ues_after"]
            )
            for row in rows
        ],
        dtype=int,
    )

    arrivals = np.asarray(
        [
            int(
                row["arrivals"]
            )
            for row in rows
        ],
        dtype=int,
    )

    session_ends = np.asarray(
        [
            int(
                row["session_ends"]
            )
            for row in rows
        ],
        dtype=int,
    )

    quality_drops = np.asarray(
        [
            int(
                row["quality_drops"]
            )
            for row in rows
        ],
        dtype=int,
    )

    potential_ho = np.asarray(
        [
            int(
                row["potential_ho_count"]
            )
            for row in rows
        ],
        dtype=int,
    )

    potential_pp = np.asarray(
        [
            int(
                row["potential_pingpong_count"]
            )
            for row in rows
        ],
        dtype=int,
    )

    avg_step_distance = np.asarray(
        [
            float(
                row["avg_step_distance"]
            )
            for row in rows
        ],
        dtype=float,
    )

    avg_best_snr = np.asarray(
        [
            float(
                row[
                    "avg_best_reference_snr_db"
                ]
            )
            for row in rows
        ],
        dtype=float,
    )
    
    print()
    print("=" * 72)
    print("UE MANAGER DATA QUALITY SUMMARY")
    print("=" * 72)

    print(
        f"Steps:                  {len(rows)}"
    )

    print(
        f"Population mean:        {ues_after.mean():.2f}"
    )

    print(
        f"Population std:         {ues_after.std():.2f}"
    )

    print(
        f"Population min/max:     "
        f"{ues_after.min()} / {ues_after.max()}"
    )

    print(
        f"Total arrivals:         {arrivals.sum()}"
    )

    print(
        f"Total session ends:     {session_ends.sum()}"
    )

    print(
        f"Total quality drops:    {quality_drops.sum()}"
    )

    print(
        f"Potential HO count:     {potential_ho.sum()}"
    )

    print(
        f"Potential PP count:     {potential_pp.sum()}"
    )

    print(
        f"Avg step distance:      "
        f"{np.nanmean(avg_step_distance):.2f}"
    )

    print(
        f"Avg best reference SNR: "
        f"{np.nanmean(avg_best_snr):.2f} dB"
    )

    print()
    print("=" * 72)
    print("WARNINGS")
    print("=" * 72)

    warnings = []

    if ues_after.mean() < 45 or ues_after.mean() > 55:
        warnings.append(
            "Population mean is too far from target 50."
        )

    if ues_after.min() < EXPECTED_MIN_UES:
        warnings.append(
            "Population sometimes becomes too low."
        )

    if ues_after.max() > EXPECTED_MAX_UES:
        warnings.append(
            "Population sometimes becomes too high."
        )

    if arrivals.sum() == 0:
        warnings.append(
            "No UE arrival was generated."
        )

    if session_ends.sum() == 0:
        warnings.append(
            "No session end was generated."
        )

    if quality_drops.sum() == 0:
        warnings.append(
            "No quality-based drop occurred. "
            "Inspect SNR distribution and drop threshold."
        )

    if potential_ho.sum() == 0:
        warnings.append(
            "No potential handover opportunity occurred."
        )

    if potential_pp.sum() == 0:
        warnings.append(
            "No potential ping-pong opportunity occurred."
        )

    if np.nanmax(avg_best_snr) > 100.0:
        warnings.append(
            "Reference SNR exceeds 100 dB. "
            "Check path loss, distance guard and noise normalization."
        )

    if warnings:
        for index, warning in enumerate(
            warnings,
            start=1,
        ):
            print(
                f"{index}. {warning}"
            )
    else:
        print(
            "No obvious issue detected."
        )


def main():
    np.random.seed(
        1
    )

    log_dir = Path(
        LOG_DIR
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    coordinates_RU = (
        build_test_ru_coordinates()
    )

    print(
        "RU coordinates:"
    )

    for ru_id, coordinate in enumerate(
        coordinates_RU
    ):
        print(
            f"RU={ru_id} | "
            f"x={coordinate[0]:8.2f} | "
            f"y={coordinate[1]:8.2f}"
        )

    manager = UEManager(
        coordinates_RU=coordinates_RU,
        target_num_UEs=NUM_INITIAL_UES,
        log_dir=log_dir,
        enable_logging=True,
    )

    manager.add_UEs_requests(
        NUM_INITIAL_UES
    )

    total_removed = {
        "quality_drop": 0,
        "session_end": 0,
        "manual_remove": 0,
        "unknown": 0,
    }

    try:
        for step in range(
            1,
            NUM_TEST_STEPS + 1,
        ):
            removed_ues, added_ids = (
                manager.UE_mobility()
            )

            removed_summary = (
                release_dummy_resources(
                    removed_ues
                )
            )

            for reason, count in (
                removed_summary.items()
            ):
                total_removed[
                    reason
                ] += count

            validate_ue_fields(
                manager
            )

            validate_population(
                num_ues=len(
                    manager.UE_requests
                ),
                step=step,
            )

            if (
                step <= 10
                or
                step % 100 == 0
            ):
                print(
                    f"[TEST] "
                    f"Step={step:04d} | "
                    f"UEs={len(manager.UE_requests):02d} | "
                    f"Added={len(added_ids):02d} | "
                    f"Removed={len(removed_ues):02d}"
                )

    finally:
        manager.close()

    print()
    print("=" * 72)
    print("REMOVED UE SUMMARY")
    print("=" * 72)

    for reason, count in (
        total_removed.items()
    ):
        print(
            f"{reason:16s}: {count}"
        )

    population_log_path = (
        find_latest_log(
            log_dir,
            "ue_population_*.csv",
        )
    )

    trajectory_log_path = (
        find_latest_log(
            log_dir,
            "ue_trajectory_*.csv",
        )
    )

    print()
    print(
        "Population log:",
        population_log_path,
    )

    print(
        "Trajectory log:",
        trajectory_log_path,
    )

    analyze_population_log(
        population_log_path
    )


if __name__ == "__main__":
    main()