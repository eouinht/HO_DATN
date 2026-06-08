import numpy as np


class A3OffsetAgent:
    """
    Baseline handover dựa trên sự kiện A3 rút gọn.

    Ý tưởng:
        - Initial attachment: chọn RU có chất lượng kênh tốt nhất.
        - UE đã được phục vụ:
            + Chỉ handover khi RU lân cận tốt nhất vượt RU hiện tại
              ít nhất (A3-Offset + hysteresis) dB.
            + Điều kiện phải duy trì liên tục đủ ttt_steps lần đánh giá.
        - Sau khi chốt RU:
            + Nếu giữ nguyên RU: giữ nguyên DU/CU.
            + Nếu initial attachment hoặc handover: chọn DU/CU khả dụng
              có năng lực xử lý còn lại lớn.
            + Chọn cặp (RB, power) nhỏ nhất thỏa throughput, latency
              và ngân sách tài nguyên hiện tại.

    Lưu ý:
        - state["UE_requests"][ue_id]["gain"] trong env_sim.py là channel gain
          đã chuẩn hóa theo noise power của một RB.
        - Vì môi trường chưa mô hình hóa RSRP riêng, baseline này dùng
          10*log10(gain) làm proxy cho chất lượng liên kết khi kiểm tra A3.
        - ttt_steps là số lần đánh giá liên tiếp, chưa phải mili giây.
          Khi mô phỏng có time-slot rõ ràng, có thể quy đổi:
              ttt_steps = ceil(TTT_ms / step_duration_ms)
    """

    def __init__(
        self,
        env,
        a3_offset_db=3.0,
        hysteresis_db=0.5,
        ttt_steps=2,
    ):
        self.env = env

        self.num_RUs = int(env.num_RUs)
        self.num_DUs = int(env.num_DUs)
        self.num_CUs = int(env.num_CUs)

        self.max_RBs_per_UE = int(env.resource_manager.max_RBs_per_UE)
        self.power_levels = np.asarray(
            env.resource_manager.P_ib_sk_val,
            dtype=float,
        )

        if self.power_levels.size == 0:
            raise ValueError("P_ib_sk_val must contain at least one power level.")

        self.power_levels = np.sort(self.power_levels)

        self.a3_offset_db = float(a3_offset_db)
        self.hysteresis_db = float(hysteresis_db)
        self.ttt_steps = max(1, int(ttt_steps))

        # Bộ nhớ A3 theo từng UE
        self.a3_candidate_ru = {}
        self.a3_counter = {}

    # =========================================================
    # Helpers
    # =========================================================
    def _get_active_ues(self, state):
        return [
            (int(ue_id), ue)
            for ue_id, ue in state["UE_requests"].items()
            if int(ue.get("status", {}).get("active", 0)) == 1
        ]

    def _quality_db(self, ue):
        """
        Proxy chất lượng liên kết theo dB.
        gain đã được chuẩn hóa theo noise power_RB trong môi trường.
        """
        gain = np.asarray(ue.get("gain", []), dtype=float)
        if gain.size != self.num_RUs:
            return np.full(self.num_RUs, -np.inf, dtype=float)

        gain = np.maximum(gain, 1e-30)
        return 10.0 * np.log10(gain)

    def _reset_a3_memory(self, ue_id):
        self.a3_candidate_ru[ue_id] = None
        self.a3_counter[ue_id] = 0

    def _update_a3_condition(self, ue_id, serving_ru, candidate_ru, quality_db):
        """
        Trả về True nếu điều kiện A3 đã được duy trì đủ ttt_steps.
        """
        if candidate_ru == serving_ru:
            self._reset_a3_memory(ue_id)
            return False

        margin_db = self.a3_offset_db + self.hysteresis_db
        condition_met = (
            quality_db[candidate_ru]
            > quality_db[serving_ru] + margin_db
        )

        if not condition_met:
            self._reset_a3_memory(ue_id)
            return False

        if self.a3_candidate_ru.get(ue_id) == candidate_ru:
            self.a3_counter[ue_id] = self.a3_counter.get(ue_id, 0) + 1
        else:
            self.a3_candidate_ru[ue_id] = candidate_ru
            self.a3_counter[ue_id] = 1

        return self.a3_counter[ue_id] >= self.ttt_steps

    def _effective_remaining_resources(self, state, ue):
        """
        env.step() sẽ giải phóng allocation cũ trước khi kiểm tra allocation mới.
        Vì vậy, khi agent tự tìm cấu hình khả thi, cần cộng lại allocation cũ
        để tránh kiểm tra quá bảo thủ.
        """
        ran = state["RAN"]

        rb_remaining = int(ran["RB_remaining"])
        ru_power_remaining = np.asarray(
            ran["RU_power_remaining"],
            dtype=float,
        ).copy()
        du_remaining = np.asarray(
            ran["DU_remaining"],
            dtype=float,
        ).copy()
        cu_remaining = np.asarray(
            ran["CU_remaining"],
            dtype=float,
        ).copy()

        alloc = ue.get("allocation", {})
        if alloc.get("RU") is not None:
            rb_remaining += int(alloc.get("num_RB_alloc", 0))
            ru_power_remaining[int(alloc["RU"])] += float(
                alloc.get("power_alloc", 0.0)
            )
            du_remaining[int(alloc["DU"])] += float(
                alloc.get("cpu_DU_req", 0.0)
            )
            cu_remaining[int(alloc["CU"])] += float(
                alloc.get("cpu_CU_req", 0.0)
            )

        return (
            rb_remaining,
            ru_power_remaining,
            du_remaining,
            cu_remaining,
        )

    def _candidate_du_cu_pairs(self, state, du_remaining, cu_remaining):
        """
        Sắp xếp các cặp DU/CU theo tổng tài nguyên xử lý còn lại giảm dần.
        Chỉ giữ các cặp có liên kết DU-CU khả dụng.
        """
        ran = state["RAN"]
        link_du_cu = np.asarray(
            ran.get("link_bw_du_cu_bps", []),
            dtype=float,
        )

        pairs = []
        for du in range(self.num_DUs):
            for cu in range(self.num_CUs):
                if link_du_cu.size > 0 and link_du_cu[du, cu] <= 0:
                    continue

                score = float(du_remaining[du] + cu_remaining[cu])
                pairs.append((score, du, cu))

        pairs.sort(reverse=True)
        return [(du, cu) for _, du, cu in pairs]

    def _ru_has_link_to_du(self, state, ru, du):
        link_ru_du = np.asarray(
            state["RAN"].get("link_bw_ru_du_bps", []),
            dtype=float,
        )

        if link_ru_du.size == 0:
            return True

        return bool(link_ru_du[ru, du] > 0)

    def _find_feasible_allocation(
        self,
        state,
        ue,
        ru_choice,
        preferred_du=None,
        preferred_cu=None,
    ):
        """
        Tìm allocation khả thi với RB và power nhỏ nhất.

        Thứ tự ưu tiên:
            1. Nếu có preferred DU/CU, thử cặp đó trước.
            2. Sau đó thử các cặp DU/CU có tài nguyên xử lý còn lại lớn.
            3. RB tăng dần, power tăng dần để tránh cấp phát dư thừa.
        """
        (
            rb_remaining,
            ru_power_remaining,
            du_remaining,
            cu_remaining,
        ) = self._effective_remaining_resources(state, ue)

        if rb_remaining <= 0:
            return None

        du_cu_pairs = []

        if preferred_du is not None and preferred_cu is not None:
            du_cu_pairs.append((int(preferred_du), int(preferred_cu)))

        for pair in self._candidate_du_cu_pairs(
            state,
            du_remaining,
            cu_remaining,
        ):
            if pair not in du_cu_pairs:
                du_cu_pairs.append(pair)

        slice_max_RBs = int(
            ue.get(
                "max_RBs",
                self.max_RBs_per_UE,
            )
        )

        rb_max = min(
            slice_max_RBs,
            rb_remaining,
        )
        for du_choice, cu_choice in du_cu_pairs:
            if not self._ru_has_link_to_du(state, ru_choice, du_choice):
                continue

            for num_rb in range(1, rb_max + 1):
                for power in self.power_levels:
                    if power > ru_power_remaining[ru_choice] + 1e-12:
                        continue

                    throughput, latency, cpu_du_req, cpu_cu_req = (
                        self.env.compute_resource_allocation(
                            ue,
                            ru_choice,
                            num_rb,
                            float(power),
                        )
                    )

                    if throughput + 1e-9 < float(ue["R_min"]):
                        continue

                    if latency > float(ue["delay"]):
                        continue

                    if cpu_du_req > du_remaining[du_choice] + 1e-9:
                        continue

                    if cpu_cu_req > cu_remaining[cu_choice] + 1e-9:
                        continue

                    return (
                        int(du_choice),
                        int(cu_choice),
                        int(num_rb),
                        float(power),
                    )

        return None

    def _fallback_action(self, ue_id, ue):
        """
        Trả về một action hợp lệ về mặt định dạng khi không tìm được
        allocation thỏa QoS. env.step() sẽ kiểm tra và reject nếu cần.
        """
        alloc = ue.get("allocation", {})

        if alloc.get("RU") is not None:
            return (
                int(ue_id),
                0,
                int(alloc["RU"]),
                int(alloc["DU"]),
                int(alloc["CU"]),
                max(1, int(alloc.get("num_RB_alloc", 1))),
                float(alloc.get("power_alloc", self.power_levels[0])),
            )

        return (
            int(ue_id),
            0,
            0,
            0,
            0,
            1,
            float(self.power_levels[0]),
        )

    # =========================================================
    # Main policy
    # =========================================================
    def select_action(self, state):
        active_ues = self._get_active_ues(state)
        if not active_ues:
            return None

        # Giữ cách chọn UE ngẫu nhiên để công bằng với baseline hiện tại
        ue_id, ue = active_ues[np.random.randint(len(active_ues))]

        quality_db = self._quality_db(ue)
        if quality_db.size == 0 or np.all(~np.isfinite(quality_db)):
            return self._fallback_action(ue_id, ue)

        alloc = ue.get("allocation", {})
        prev_ru = alloc.get("RU")
        prev_du = alloc.get("DU")
        prev_cu = alloc.get("CU")

        best_ru = int(np.argmax(quality_db))

        # =====================================================
        # CASE 1: Initial attachment
        # =====================================================
        if prev_ru is None:
            self._reset_a3_memory(ue_id)

            ru_order = list(np.argsort(-quality_db))
            for ru_choice in ru_order:
                found = self._find_feasible_allocation(
                    state,
                    ue,
                    int(ru_choice),
                )

                if found is not None:
                    du_choice, cu_choice, num_rb, power = found
                    return (
                        int(ue_id),
                        0,
                        int(ru_choice),
                        int(du_choice),
                        int(cu_choice),
                        int(num_rb),
                        float(power),
                    )

            return self._fallback_action(ue_id, ue)

        # =====================================================
        # CASE 2: UE đã có mapping
        # =====================================================
        prev_ru = int(prev_ru)
        prev_du = int(prev_du)
        prev_cu = int(prev_cu)

        should_handover = self._update_a3_condition(
            ue_id,
            prev_ru,
            best_ru,
            quality_db,
        )

        # -----------------------------------------------------
        # 2.1. Giữ nguyên mapping nếu chưa đủ điều kiện A3/TTT
        # -----------------------------------------------------
        if not should_handover:
            found = self._find_feasible_allocation(
                state,
                ue,
                prev_ru,
                preferred_du=prev_du,
                preferred_cu=prev_cu,
            )

            if found is not None:
                du_choice, cu_choice, num_rb, power = found

                # env.step() yêu cầu keep mapping phải giữ nguyên DU/CU
                # nên chỉ dùng allocation nếu cặp DU/CU vẫn là cặp cũ.
                if du_choice == prev_du and cu_choice == prev_cu:
                    return (
                        int(ue_id),
                        0,
                        prev_ru,
                        prev_du,
                        prev_cu,
                        int(num_rb),
                        float(power),
                    )

            return self._fallback_action(ue_id, ue)

        # -----------------------------------------------------
        # 2.2. Handover sang RU ứng viên
        # -----------------------------------------------------
        found = self._find_feasible_allocation(
            state,
            ue,
            best_ru,
        )

        if found is not None:
            du_choice, cu_choice, num_rb, power = found

            # Sau khi HO thành công, reset bộ đếm A3
            self._reset_a3_memory(ue_id)

            return (
                int(ue_id),
                1,
                int(best_ru),
                int(du_choice),
                int(cu_choice),
                int(num_rb),
                float(power),
            )

        # Nếu RU ứng viên chưa đủ tài nguyên, tiếp tục giữ mapping cũ
        self._reset_a3_memory(ue_id)

        found = self._find_feasible_allocation(
            state,
            ue,
            prev_ru,
            preferred_du=prev_du,
            preferred_cu=prev_cu,
        )

        if found is not None:
            du_choice, cu_choice, num_rb, power = found
            if du_choice == prev_du and cu_choice == prev_cu:
                return (
                    int(ue_id),
                    0,
                    prev_ru,
                    prev_du,
                    prev_cu,
                    int(num_rb),
                    float(power),
                )

        return self._fallback_action(ue_id, ue)
