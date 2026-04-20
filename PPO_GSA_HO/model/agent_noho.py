import numpy as np


class NoHandoverAgent:
    def __init__(self, env):
        self.num_RUs = int(env.num_RUs)
        self.num_DUs = int(env.num_DUs)
        self.num_CUs = int(env.num_CUs)
        self.max_RBs_per_UE = int(env.resource_manager.max_RBs_per_UE)

        self.power_levels = np.asarray(env.resource_manager.P_ib_sk_val, dtype=float)
        if self.power_levels.size == 0:
            self.power_levels = np.asarray([0.0], dtype=float)

    def select_action(self, state):
        active_ues = [
            (int(uid), ue)
            for uid, ue in state["UE_requests"].items()
            if int(ue.get("status", {}).get("active", 0)) == 1
        ]

        if not active_ues:
            return None

        ue_id, ue = active_ues[np.random.randint(len(active_ues))]
        alloc = ue.get("allocation", {})

        # =========================
        # CASE 1: đã ACCEPT → GIỮ NGUYÊN hoàn toàn
        # =========================
        if alloc.get("RU") is not None:
            return (
                ue_id,
                0,  # no handover
                int(alloc["RU"]),
                int(alloc["DU"]),
                int(alloc["CU"]),
                int(alloc.get("num_RB_alloc", np.random.randint(1, self.max_RBs_per_UE + 1))),
                float(alloc.get("power_alloc", np.random.choice(self.power_levels))),
            )

        # =========================
        # CASE 2: CHƯA ACCEPT → gán mới mapping + resource
        # =========================
        return (
            ue_id,
            0,
            int(np.random.randint(0, self.num_RUs)),
            int(np.random.randint(0, self.num_DUs)),
            int(np.random.randint(0, self.num_CUs)),
            int(np.random.randint(1, self.max_RBs_per_UE + 1)),
            float(np.random.choice(self.power_levels)),
        )