from pathlib import Path
from datetime import datetime
import os
import time
import random
import numpy as np
import torch

from config import *
from env.env_sim import HandOverEnv
from model.agent_PPO_GSA_MLP import PPOAgent, load_checkpoint
from model.agent_max_gain import MaxGainAgent
from model.agent_random import RandomAgent
from model.agent_noho import NoHandoverAgent


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_result_paths(results_root, method_name):
    base = Path(results_root) / method_name
    base.mkdir(parents=True, exist_ok=True)

    return {
        "dir": base,
        "reward": base / "evaluation_reward.txt",
        "accept": base / "evaluation_accept.txt",
        "throughput": base / "evaluation_throughput.txt",
        "latency": base / "evaluation_latency.txt",
        "handover": base / "evaluation_handover.txt",
        "time": base / "evaluation_time.txt",
        "numUEs": base / "evaluation_numUEs.txt",
        "acc_term": base / "evaluation_acc_term.txt",
        "thr_term": base / "evaluation_thr_term.txt",
        "lat_term": base / "evaluation_lat_term.txt",
        "handover_term": base / "evaluation_handover_term.txt",
        "pingpong": base / "evaluation_pingpong.txt",
    }


def write_results(paths, metrics, mode="a"):
    mapping = {
        "reward": "reward",
        "accept": "accept",
        "throughput": "throughput",
        "latency": "latency",
        "handover": "handover",
        "time": "time",
        "numUEs": "num_ues",
        "acc_term": "acc_term",
        "thr_term": "thr_term",
        "lat_term": "lat_term",
        "handover_term": "ho_term",
        "pingpong": "pingpong",
    }

    for file_key, metric_key in mapping.items():
        with open(paths[file_key], mode) as f:
            f.write(f"{metrics[metric_key]}\n")


def make_env(num_UEs, num_RBs, num_RUs, num_DUs, num_CUs, seed):
    set_global_seed(seed)
    total_nodes = num_RUs + num_DUs + num_CUs
    return HandOverEnv(num_UEs, num_RBs, total_nodes, num_RUs, num_DUs, num_CUs)


def get_env_action(agent, state):
    out = agent.select_action(state)

    if isinstance(out, tuple) and len(out) == 4:
        _, cached_action, _, _ = out
        if cached_action is None:
            return None
        return cached_action.env_action

    return out


def run_one_episode(agent, env):
    num_ues_start = len(env.UE_manager.UE_requests)

    state = env.get_state()
    total_rew = 0.0
    total_acc = 0.0
    total_thr = 0.0
    total_lat = 0.0
    total_ho = 0.0
    total_pingpong = 0.0

    acc_term = 0.0
    thr_term = 0.0
    lat_term = 0.0
    ho_term = 0.0

    done = False
    start_time = time.time()

    while not done:
        action = get_env_action(agent, state)
        if action is None:
            break

        next_state, reward, done, info = env.step(action)
        total_rew += float(reward)

        #print(f"info = {info}")

        if isinstance(info, dict) and info.get("success", False):
            total_acc += 1
            total_thr += float(info.get("throughput_UE", 0.0))
            total_lat += float(info.get("latency_UE", 0.0))
            total_ho += float(info.get("handover", 0.0))
            total_pingpong += float(info.get("pingpong", 0.0))
            acc_term += float(info.get("acc_term", 0.0))
            thr_term += float(info.get("thr_term", 0.0))
            lat_term += float(info.get("lat_term", 0.0))
            ho_term += float(info.get("handover_term", 0.0))

        state = next_state

    elapsed = time.time() - start_time

    return {
        "reward": total_rew,
        "accept": total_acc,
        "throughput": total_thr,
        "latency": total_lat,
        "handover": total_ho,
        "time": elapsed,
        "num_ues": num_ues_start,
        "acc_term": acc_term,
        "thr_term": thr_term,
        "lat_term": lat_term,
        "ho_term": ho_term,
        "pingpong": total_pingpong,
    }


def apply_same_mobility_to_all_envs(envs, mobility_seed):
    for _, env in envs.items():
        set_global_seed(mobility_seed)

        removed_ues_with_info, _ = env.UE_manager.UE_mobility()
        for _, ue_info in removed_ues_with_info:
            alloc = ue_info.get("allocation", {})
            if alloc["RU"] is not None:
                env.resource_manager.release_resources(
                    alloc["RU"],
                    alloc["DU"],
                    alloc["CU"],
                    alloc["num_RB_alloc"],
                    alloc["power_alloc"],
                    alloc["cpu_DU_req"],
                    alloc["cpu_CU_req"],
                )


def main():
    results_root = "./results_evaluation"
    os.makedirs(results_root, exist_ok=True)

    configs = [
        (50, 273, 7, 5, 5),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(results_root) / f"run_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    base_seed = 20260413
    max_episode = 5

    for num_UEs, num_RBs, num_RUs, num_DUs, num_CUs in configs:
        print("\n==============================")
        print(f"🚀 Chạy thử với {num_UEs} UEs")
        print("==============================")

        paths_PPOGSA = build_result_paths(run_root, "PPOGSA")
        paths_RANDOM = build_result_paths(run_root, "RANDOM")
        paths_MAXGAIN = build_result_paths(run_root, "MAXGAIN")
        paths_NOHO = build_result_paths(run_root, "NO_HANDOVER")

        # Tạo env một lần, giữ xuyên suốt nhiều episode
        envs = {
            "PPOGSA": make_env(num_UEs, num_RBs, num_RUs, num_DUs, num_CUs, seed=base_seed),
            "RANDOM": make_env(num_UEs, num_RBs, num_RUs, num_DUs, num_CUs, seed=base_seed),
            "MAXGAIN": make_env(num_UEs, num_RBs, num_RUs, num_DUs, num_CUs, seed=base_seed),
            "NO_HANDOVER": make_env(num_UEs, num_RBs, num_RUs, num_DUs, num_CUs, seed=base_seed),
        }

        agent_PPOGSA = PPOAgent(learning_rate=0.001)
        checkpoint_path = "./results/run_PPOGSA_50UE/checkpoint_PPOGSA.pt"
        load_checkpoint(agent_PPOGSA, checkpoint_path, partial=True)

        agent_RANDOM = RandomAgent(envs["RANDOM"])
        agent_MAXGAIN = MaxGainAgent(envs["MAXGAIN"])
        agent_NOHO = NoHandoverAgent(envs["NO_HANDOVER"])

        agents = {
            "PPOGSA": (agent_PPOGSA, paths_PPOGSA),
            "RANDOM": (agent_RANDOM, paths_RANDOM),
            "MAXGAIN": (agent_MAXGAIN, paths_MAXGAIN),
            "NO_HANDOVER": (agent_NOHO, paths_NOHO),
        }

        for ep in range(1, max_episode + 1):
            print(f"\n--- Episode {ep}/{max_episode} ---")

            for method_name, (agent, paths) in agents.items():
                metrics = run_one_episode(agent, envs[method_name])
                write_results(paths, metrics, mode="a")

                avg_thr = metrics["throughput"] / max(metrics["accept"], 1) / 1e6
                avg_lat = metrics["latency"] / max(metrics["accept"], 1) * 1e3
                ho_rate = metrics["handover"] / max(metrics["accept"], 1)
                pp_rate = metrics["pingpong"] / max(metrics["accept"], 1)

                print(
                    f"[{method_name}] "
                    f"Reward={metrics['reward']:.2f} | "
                    f"UEs={metrics['num_ues']:.0f} | "
                    f"Accept={metrics['accept']:.0f} | "
                    f"Thr={metrics['throughput']/1e6:.2f}({avg_thr:.2f})Mb | "
                    f"Lat={metrics['latency']*1e3:.2f}({avg_lat:.2f})ms | "
                    f"HO={metrics['handover']:.0f}({ho_rate:.2f}) | "
                    f"PP={metrics['pingpong']:.0f}({pp_rate:.2f})"
                )

            # Mobility sau khi tất cả agent chạy xong episode hiện tại
            mobility_seed = base_seed + 100000 + ep
            apply_same_mobility_to_all_envs(envs, mobility_seed)

            

if __name__ == "__main__":
    set_global_seed(1)
    main()