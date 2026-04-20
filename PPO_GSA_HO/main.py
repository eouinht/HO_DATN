import os
import time
import random
import numpy as np
from pathlib import Path
from config import *
from datetime import datetime
from env.env_sim import HandOverEnv
from model.agent_PPO_GSA_MLP import PPOAgent, FullPolicy, train_agent, evaluate_agent, save_checkpoint, load_checkpoint


# -----------------------------
# 1. HUẤN LUYỆN PPO (train)
# -----------------------------
def train_ppogsa(num_UEs, num_RBs, total_nodes, num_RUs, num_DUs, num_CUs, results_dir):
    """
    Huấn luyện PPO và lưu checkpoint cuối.
    Trả về đường dẫn checkpoint.
    """

    learning_rate = 0.001
    agent_ppogsa = PPOAgent(
        learning_rate=learning_rate
    )

    env = HandOverEnv(num_UEs, num_RBs, total_nodes, num_RUs, num_DUs, num_CUs)

    print("🚀 Bắt đầu huấn luyện PPO agent ...")
    agent_ppogsa_trained = train_agent(env, agent_ppogsa, results_dir)

    checkpoint_path = os.path.join(results_dir, "checkpoint_PPOGSA.pt")
    save_checkpoint(agent_ppogsa_trained, checkpoint_path)
    return checkpoint_path


def main():

    os.makedirs(results_root, exist_ok=True)

    configs = [
        (50, 273, 7, 5, 5),
    ]

    for num_UEs, num_RBs, num_RUs, num_DUs, num_CUs in configs:

        print("\n==============================")
        print(f"🚀 Chạy thử với {num_UEs} UEs")
        print("==============================")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_dir = os.path.join(results_root, f"run_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # ---- Train 1 lần ----

        total_nodes = num_RUs + num_DUs + num_CUs
        checkpoint_path = train_ppogsa(num_UEs, num_RBs, total_nodes, num_RUs, num_DUs, num_CUs, results_dir)
        #train_maxgain(num_UEs, num_RBs, total_nodes, num_RUs, num_DUs, num_CUs, results_dir)
        #checkpoint_path = r"D:\Research\Handover Problem\Code\PPO_GSA_HO\results\run_20260407_134412\checkpoint_PPOGSA.pt"
        #evaluate_ppogsa(num_UEs, num_RBs, total_nodes, checkpoint_path, results_dir, num_RUs, num_DUs, num_CUs)


results_root = "./results"

if __name__ == "__main__":
    # seed
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    main()

