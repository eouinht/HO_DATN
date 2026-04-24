from typing import List

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import UE, Cell, HandoverEvent


class HandoverEngine:

    def check(
        self,
        ues: List[UE],
        cells: List[Cell],
        cfg: SimulationConfig,
        current_time: float,
        rng_manager: RNGManager,
    ) -> tuple[List[HandoverEvent], List[UE]]:

        events: List[HandoverEvent] = []
        time_key = int(current_time * 1000)

        for ue in ues:
            if not ue.connected or ue.serving_cell is None:
                ue.ho_timer_s = 0.0
                continue

            serving_meas = self._get_serving_measurement(ue)

            if serving_meas is None:
                ue.ho_timer_s = 0.0
                continue

            target = self._find_best_target(ue, serving_meas, cfg)

            if target is None:
                ue.ho_timer_s = 0.0
                continue

            # A3 condition đã thỏa → tăng TTT
            ue.ho_timer_s += cfg.time_step

            if ue.ho_timer_s < (cfg.default_ttt_ms / 1000.0):
                continue

            # Thực hiện HO
            rng = rng_manager.get_rng_for(
                ue.id,
                int(current_time * 1000),
            )

            ho_success = self._evaluate_handover_success(
                ue,
                target,
                cfg,
                rng,
            )

            event = HandoverEvent(
                ue_id=ue.id,
                cell_source=ue.serving_cell,
                cell_target=target.cell_id,

                ru_source=ue.serving_ru,
                ru_target=target.ru_id,

                rsrp_source=serving_meas.rsrp,
                rsrp_target=target.rsrp,

                rsrq_source=serving_meas.rsrq,
                rsrq_target=target.rsrq,

                sinr_source=serving_meas.sinr,
                sinr_target=target.sinr,

                a3_offset_db=cfg.default_a3_offset,
                ttt_ms=cfg.default_ttt_ms,

                ho_success=ho_success,
                timestamp_s=current_time,
            )

            if ho_success:
                self._apply_handover(ue, target)
            else:
                ue.drop_count += 1

            ue.ho_timer_s = 0.0
            events.append(event)

        return events, ues

    def _get_serving_measurement(self, ue: UE):
        for m in ue.neighbor_measurements:
            if m.cell_id == ue.serving_cell:
                return m
        return None

    def _find_best_target(self, ue: UE, serving_meas, cfg):
        candidates = []

        for m in ue.neighbor_measurements:
            if m.cell_id == ue.serving_cell:
                continue

            # A3 condition:
            if m.rsrp > serving_meas.rsrp + cfg.default_a3_offset:
                candidates.append(m)

        if not candidates:
            return None

        return max(candidates, key=lambda x: x.rsrp)

    def _evaluate_handover_success(
        self,
        ue: UE,
        target,
        cfg: SimulationConfig,
        rng,
    ) -> bool:
        """
        HO success model:
        - phụ thuộc SINR target
        - thêm random nhẹ
        """

        base_prob = 0.9

        if target.sinr < cfg.embb_min_sinr_db:
            base_prob -= 0.2

        if target.sinr < 0:
            base_prob -= 0.3

        base_prob = max(0.1, min(0.99, base_prob))

        return rng.random() < base_prob

    def _apply_handover(self, ue: UE, target) -> None:
        ue.serving_cell = target.cell_id
        ue.serving_ru = target.ru_id

        ue.rsrp = target.rsrp
        ue.rsrq = target.rsrq
        ue.sinr = target.sinr

        ue.handover_count += 1