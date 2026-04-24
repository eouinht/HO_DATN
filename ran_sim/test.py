# test.py

from collections import Counter

from core.config import SimulationConfig
from rl.gym_env import RANGymEnv


def make_cfg(
    sim_time=10.0,
    num_ues=10,
    enable_dynamic_ues=False,
):
    return SimulationConfig(
        sim_time=sim_time,
        time_step=1.0,
        num_ues=num_ues,

        min_ues=10,
        max_ues=50,
        enable_dynamic_ues=enable_dynamic_ues,
    )


def test_env_reset():
    cfg = make_cfg(sim_time=10.0, num_ues=10)

    env = RANGymEnv(cfg)
    obs, info = env.reset(seed=42)

    print("OBS shape:", obs.shape)
    print("INFO:", info)

    assert obs is not None
    assert obs.shape == env.observation_space.shape


def test_sim_core_step():
    cfg = make_cfg(sim_time=10.0, num_ues=10)

    env = RANGymEnv(cfg)
    obs, info = env.reset(seed=42)

    state = env.sim.get_state()

    action = None

    for ue in state["ues"]:
        if ue["connected"] and ue["serving_cell"] is not None:
            action = {
                "ue_id": ue["id"],
                "cell_id": ue["serving_cell"],
                "du_id": 1,
                "cu_id": 1,
                "num_prbs": 10,
                "tx_power_watts": 20.0,
            }
            break

    assert action is not None, "No connected UE found after reset"

    state, reward, done, info = env.sim.step(action)

    print("\n========== CORE STEP TEST ==========")
    print("Action:", action)
    print("Reward:", reward)
    print("Info:", info)
    print("====================================\n")

    assert isinstance(state, dict)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_gym_step():
    cfg = make_cfg(sim_time=10.0, num_ues=10)

    env = RANGymEnv(cfg)
    obs, info = env.reset(seed=42)

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print("\n========== GYM STEP TEST ==========")
    print("Gym action:", action)
    print("Reward:", reward)
    print("Info:", info)
    print("===================================\n")

    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_episode_stats():
    cfg = make_cfg(
        sim_time=300.0,
        num_ues=10,
        enable_dynamic_ues=False,
    )

    env = RANGymEnv(cfg)
    obs, info = env.reset(seed=42)

    reason_counter = Counter()
    success_counter = Counter()

    reward_sum = 0.0
    throughput_list = []
    latency_list = []

    steps = int(cfg.sim_time / cfg.time_step)

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        reward_sum += reward

        success = bool(info.get("success", False))
        reason = info.get("reason", "unknown")

        success_counter["success" if success else "fail"] += 1
        reason_counter[reason] += 1

        if "throughput_bps" in info:
            throughput_list.append(info["throughput_bps"] / 1e6)

        if "latency_s" in info:
            latency_list.append(info["latency_s"] * 1000)

        assert obs.shape == env.observation_space.shape

        if terminated or truncated:
            break

    print("\n========== EPISODE SUMMARY ==========")
    print(f"Steps: {step + 1}")
    print(f"Total reward: {reward_sum:.3f}")
    print(f"Avg reward: {reward_sum / (step + 1):.3f}")

    print("\n--- Success / Fail ---")
    for k, v in success_counter.items():
        print(f"{k}: {v}")

    print("\n--- Reason distribution ---")
    for r, c in reason_counter.most_common():
        print(f"{r}: {c}")

    if throughput_list:
        print("\n--- Throughput (Mbps) ---")
        print(f"Avg: {sum(throughput_list) / len(throughput_list):.2f}")
        print(f"Min: {min(throughput_list):.2f}")
        print(f"Max: {max(throughput_list):.2f}")

    if latency_list:
        print("\n--- Latency (ms) ---")
        print(f"Avg: {sum(latency_list) / len(latency_list):.2f}")
        print(f"Min: {min(latency_list):.2f}")
        print(f"Max: {max(latency_list):.2f}")

    print("=====================================\n")


def test_ue_lifecycle():
    cfg = make_cfg(
        sim_time=20.0,
        num_ues=50,
        enable_dynamic_ues=True,
    )

    env = RANGymEnv(cfg)
    obs, info = env.reset(seed=42)

    print("\n========== UE LIFECYCLE TEST ==========")
    print("Initial UE count:", len(env.sim.ues))
    print("Initial OBS shape:", obs.shape)

    assert cfg.min_ues <= len(env.sim.ues) <= cfg.max_ues
    assert obs.shape == env.observation_space.shape

    for step in range(int(cfg.sim_time / cfg.time_step)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Step {step + 1}: "
            f"UE count={len(env.sim.ues)}, "
            f"arrivals={info.get('num_arrivals', 0)}, "
            f"departures={info.get('num_departures', 0)}, "
            f"reward={reward:.3f}, "
            f"reason={info.get('reason', 'unknown')}"
        )

        assert cfg.min_ues <= len(env.sim.ues) <= cfg.max_ues
        assert obs.shape == env.observation_space.shape

        if terminated or truncated:
            break

    print("=======================================\n")


if __name__ == "__main__":
    test_env_reset()
    test_sim_core_step()
    test_gym_step()
    test_episode_stats()
    test_ue_lifecycle()

    print("All tests passed.")