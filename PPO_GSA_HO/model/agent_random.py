import numpy as np


class RandomAgent:
    def __init__(self, env):
        self.env = env

        self.num_RUs = int(env.num_RUs)
        self.num_DUs = int(env.num_DUs)
        self.num_CUs = int(env.num_CUs)

        self.max_RBs_per_UE = int(env.resource_manager.max_RBs_per_UE)
        self.power_levels = np.asarray(env.resource_manager.P_ib_sk_val, dtype=float)

        if self.power_levels.size == 0:
            self.power_levels = np.asarray([0.0], dtype=float)

    def select_action(self, state):
        # =========================
        # Lọc UE active
        # =========================
        active_ues = [
            (int(ue_id), ue)
            for ue_id, ue in state["UE_requests"].items()
            if int(ue.get("status", {}).get("active", 0)) == 1
        ]

        if len(active_ues) == 0:
            return None

        ue_id, ue = active_ues[np.random.randint(0, len(active_ues))]

        prev_alloc = ue.get("allocation", {})
        prev_ru = prev_alloc.get("RU")
        prev_du = prev_alloc.get("DU")
        prev_cu = prev_alloc.get("CU")

        has_prev = prev_ru is not None

        # =========================
        # Handover decision
        # =========================
        if has_prev:
            handover_flag = int(np.random.rand() < 0.5)

            if handover_flag == 0:
                ru_choice = int(prev_ru)
                du_choice = int(prev_du)
                cu_choice = int(prev_cu)
            else:
                ru_choice = int(np.random.randint(0, self.num_RUs))
                du_choice = int(np.random.randint(0, self.num_DUs))
                cu_choice = int(np.random.randint(0, self.num_CUs))
        else:
            handover_flag = 0
            ru_choice = int(np.random.randint(0, self.num_RUs))
            du_choice = int(np.random.randint(0, self.num_DUs))
            cu_choice = int(np.random.randint(0, self.num_CUs))

        # =========================
        # Resource allocation
        # =========================
        num_RB_alloc = int(np.random.randint(1, self.max_RBs_per_UE + 1))
        power_alloc = float(np.random.choice(self.power_levels))

        return (
            int(ue_id),
            int(handover_flag),
            int(ru_choice),
            int(du_choice),
            int(cu_choice),
            int(num_RB_alloc),
            float(power_alloc),
        )