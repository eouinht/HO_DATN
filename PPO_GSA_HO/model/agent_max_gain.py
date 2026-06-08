import numpy as np


class MaxGainAgent:
    def __init__(self, env):
        self.env = env

        self.num_RUs = int(env.num_RUs)
        self.num_DUs = int(env.num_DUs)
        self.num_CUs = int(env.num_CUs)

        # Giới hạn lớn nhất toàn hệ thống, ví dụ 20 RB.
        # Giá trị thực tế cho từng UE sẽ lấy theo slice.
        self.max_RBs_per_UE = int(
            env.resource_manager.max_RBs_per_UE
        )

        self.P_ib_sk_val = np.asarray(
            env.resource_manager.P_ib_sk_val,
            dtype=float,
        )

        if self.P_ib_sk_val.size == 0:
            self.P_ib_sk_val = np.asarray(
                [0.0],
                dtype=float,
            )

    def _get_active_ues(self, state):
        return [
            (int(uid), ue)
            for uid, ue in state["UE_requests"].items()
            if int(
                ue.get(
                    "status",
                    {},
                ).get(
                    "active",
                    0,
                )
            ) == 1
        ]

    def _get_slice_max_rbs(self, ue):
        """
        Trả về số RB tối đa theo slice.

        Ví dụ:
            eMBB  -> 20 RB
            uRLLC -> 10 RB

        Nếu UE chưa có trường max_RBs thì fallback về
        giới hạn lớn nhất toàn hệ thống.
        """
        return int(
            ue.get(
                "max_RBs",
                self.max_RBs_per_UE,
            )
        )

    def select_action(self, state):
        active_ues = self._get_active_ues(
            state
        )

        if not active_ues:
            return None

        # =========================
        # Pick random UE
        # Giữ giống các baseline khác
        # =========================
        ue_id, ue = active_ues[
            np.random.randint(
                len(active_ues)
            )
        ]

        gain = np.asarray(
            ue.get(
                "gain",
                [],
            ),
            dtype=float,
        )

        if gain.size == 0:
            return None

        alloc = ue.get(
            "allocation",
            {},
        )

        prev_ru = alloc.get("RU")
        prev_du = alloc.get("DU")
        prev_cu = alloc.get("CU")

        # =========================
        # Best RU by channel gain
        # =========================
        best_ru = int(
            np.argmax(
                gain
            )
        )

        # =========================
        # Handover logic
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
            du_choice = int(
                np.random.randint(
                    0,
                    self.num_DUs,
                )
            )

            cu_choice = int(
                np.random.randint(
                    0,
                    self.num_CUs,
                )
            )
        else:
            du_choice = int(prev_du)
            cu_choice = int(prev_cu)

        # =========================
        # RB allocation theo slice
        # =========================
        slice_max_rbs = self._get_slice_max_rbs(
            ue
        )

        avg_rbs_per_ue = max(
            1,
            int(
                self.env.num_RBs
                /
                max(
                    self.env.num_UEs,
                    1,
                )
            ),
        )

        num_RB_alloc = min(
            slice_max_rbs,
            avg_rbs_per_ue,
        )

        # =========================
        # Power allocation
        # =========================
        power_alloc = float(
            np.random.choice(
                self.P_ib_sk_val
            )
        )

        return (
            int(ue_id),
            int(handover_flag),
            int(ru_choice),
            int(du_choice),
            int(cu_choice),
            int(num_RB_alloc),
            float(power_alloc),
        )