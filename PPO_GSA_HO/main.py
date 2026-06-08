import os
import random
import numpy as np
import torch
from datetime import datetime

from config import *
from env.env_sim import HandOverEnv
from model.agent_PPO_GSA_MLP import (
    PPOAgent,
    train_agent,
    save_checkpoint,
)


# =========================================================
# Seed
# =========================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Train PPO
# =========================================================
def train_ppogsa(
    num_UEs,
    num_RBs,
    total_nodes,
    num_RUs,
    num_DUs,
    num_CUs,
    results_dir,
    max_episode,
):
    """
    Huấn luyện PPO và lưu checkpoint cuối.

    Trả về:
        checkpoint_path: đường dẫn checkpoint đã lưu
    """
    learning_rate = 0.001

    agent_ppogsa = PPOAgent(
        learning_rate=learning_rate,
    )
    radio_log_path = os.path.join(
        results_dir,
        "radio_metrics.csv",
    )
    
    env = HandOverEnv(
        num_UEs,
        num_RBs,
        total_nodes,
        num_RUs,
        num_DUs,
        num_CUs,
        radio_log_path=radio_log_path
    )

    # Lưu cấu hình huấn luyện
    config_path = os.path.join(results_dir, "train_config.txt")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(f"num_UEs={num_UEs}\n")
        f.write(f"num_RBs={num_RBs}\n")
        f.write(f"num_RUs={num_RUs}\n")
        f.write(f"num_DUs={num_DUs}\n")
        f.write(f"num_CUs={num_CUs}\n")
        f.write(f"max_RBs_per_UE={max_RBs_per_UE}\n")
        f.write(f"learning_rate={learning_rate}\n")
        f.write(f"max_episode={max_episode}\n")

    print("🚀 Bắt đầu huấn luyện PPO agent ...")

    agent_ppogsa_trained = train_agent(
        env,
        agent_ppogsa,
        results_dir,
        max_episode=max_episode,
    )
    env.close()
    checkpoint_path = os.path.join(
        results_dir,
        "checkpoint_PPOGSA.pt",
    )

    save_checkpoint(
        agent_ppogsa_trained,
        checkpoint_path,
    )

    return checkpoint_path


# =========================================================
# Main
# =========================================================
def main():
    results_root = "./results"
    os.makedirs(results_root, exist_ok=True)

    configs = [
        (50, 273, 7, 5, 5),
    ]

    # Test nhanh: 20
    # Train chính thức: 50000
    max_episode = 100

    for num_UEs, num_RBs, num_RUs, num_DUs, num_CUs in configs:
        print("\n==============================")
        print(f"🚀 Huấn luyện với {num_UEs} UEs")
        print("==============================")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_dir = os.path.join(
            results_root,
            f"run_PPOGSA_{num_UEs}UE_{timestamp}",
        )

        os.makedirs(results_dir, exist_ok=True)

        total_nodes = num_RUs + num_DUs + num_CUs

        checkpoint_path = train_ppogsa(
            num_UEs,
            num_RBs,
            total_nodes,
            num_RUs,
            num_DUs,
            num_CUs,
            results_dir,
            max_episode=max_episode,
        )

        print(f"✅ Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    set_global_seed(1)
    main()