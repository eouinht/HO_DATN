import numpy as np


class MaxGainAgent:
    def __init__(self, env):
        self.env = env

        self.num_RUs = int(env.num_RUs)
        self.num_DUs = int(env.num_DUs)
        self.num_CUs = int(env.num_CUs)

        self.max_RBs_per_UE = int(env.resource_manager.max_RBs_per_UE)

        self.P_ib_sk_val = np.asarray(env.resource_manager.P_ib_sk_val, dtype=float)

    def _get_active_ues(self, state):
        return [
            (int(uid), ue)
            for uid, ue in state["UE_requests"].items()
            if int(ue.get("status", {}).get("active", 0)) == 1
        ]

    def select_action(self, state):
        active_ues = self._get_active_ues(state)
        if not active_ues:
            return None

        # =========================
        # pick random UE (baseline fairness)
        # =========================
        ue_id, ue = active_ues[np.random.randint(len(active_ues))]

        gain = np.asarray(ue.get("gain", []), dtype=float)
        if gain.size == 0:
            return None

        alloc = ue.get("allocation", {})

        prev_ru = alloc.get("RU")
        prev_du = alloc.get("DU")
        prev_cu = alloc.get("CU")

        # =========================
        # best RU by channel gain
        # =========================
        best_ru = int(np.argmax(gain))

        # =========================
        # handover logic (simple)
        # =========================
        if prev_ru is None:
            handover_flag = 0
            ru_choice = best_ru
        else:
            if best_ru != int(prev_ru):
                handover_flag = 1
                ru_choice = best_ru
            else:
                handover_flag = 0
                ru_choice = int(prev_ru)

        # =========================
        # DU / CU logic
        # =========================
        if prev_ru is None or handover_flag == 1:
            du_choice = int(np.random.randint(0, self.num_DUs))
            cu_choice = int(np.random.randint(0, self.num_CUs))
        else:
            du_choice = int(prev_du)
            cu_choice = int(prev_cu)

        # =========================
        # RB & power (greedy baseline)
        # =========================
        # num_RB_alloc = int(self.env.num_RBs / self.env.num_UEs)
        num_RB_alloc = min(
            self.max_RBs_per_UE,
            max(1, int(self.env.num_RBs / self.env.num_UEs)),
        )
        power_alloc = float(np.random.choice(self.P_ib_sk_val))

        return (
            int(ue_id),
            int(handover_flag),
            int(ru_choice),
            int(du_choice),
            int(cu_choice),
            int(num_RB_alloc),
            float(power_alloc),
        )