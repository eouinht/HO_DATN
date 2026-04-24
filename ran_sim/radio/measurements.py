# radio/measurements.py

from typing import List
import math

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import UE, Cell, NeighborMeasurement
from radio.pathloss import calculate_path_loss_db
# from radio.channel import dbm_to_watt, watt_to_dbm


class MeasurementEngine:
    """
    Cập nhật đo đạc radio cho từng UE:
    - RSRP / RSRQ: phục vụ attachment + handover A3/TTT
    - SINR: giữ dạng dB để làm KPI / điều kiện QoS
    - neighbor_measurements: danh sách cell UE có thể đo được

    Lưu ý:
    - Throughput, SNR tuyến tính, channel gain MIMO sẽ tính ở resource/allocation.py
    - File này chỉ làm measurement layer.
    """

    def update(
        self,
        ues: List[UE],
        cells: List[Cell],
        cfg: SimulationConfig,
        current_time: float,
        rng_manager: RNGManager,
    ) -> List[UE]:

        time_key = int(current_time * 1000)

        for ue in ues:
            measurements: List[NeighborMeasurement] = []

            for cell in cells:
                rng = rng_manager.get_rng_for(
                    ue.id,
                    cell.id,
                    time_key,
                )

                rsrp_dbm, rsrq_db, sinr_db = self._measure_ue_cell(
                    ue=ue,
                    cell=cell,
                    cfg=cfg,
                    rng=rng,
                )

                if rsrp_dbm >= cfg.rsrp_measurement_threshold - 5.0:
                    measurements.append(
                        NeighborMeasurement(
                            cell_id=cell.id,
                            ru_id=cell.ru_id,
                            rsrp=rsrp_dbm,
                            rsrq=rsrq_db,
                            sinr=sinr_db,
                        )
                    )

            ue.neighbor_measurements = measurements

            if not measurements:
                self._clear_radio_state(ue)
                continue

            self._update_serving_measurement(ue)

        return ues

    def _measure_ue_cell(
        self,
        ue: UE,
        cell: Cell,
        cfg: SimulationConfig,
        rng,
    ) -> tuple[float, float, float]:
        """
        Measurement model kết hợp:
        - Pathloss từ env cũ
        - RSRP/RSRQ/SINR measurement từ MATLAB mới

        RSRP:
            rsrp_dbm = tx_power_dbm - path_loss_db + shadowing

        SINR measurement:
            dùng dạng đơn giản theo dB:
            sinr_db = rsrp_dbm - noise_power_dbm + random_noise

        Throughput/SNR thật theo Rayleigh MIMO không tính ở đây.
        """

        distance_m = math.hypot(
            ue.x - cell.x,
            ue.y - cell.y,
        )
        distance_m = max(distance_m, 1.0)

        path_loss_db = calculate_path_loss_db(
            distance_m=distance_m,
            carrier_frequency_hz=cfg.carrier_frequency,
        )

        shadowing_db = rng.gauss(0.0, cfg.shadowing_std_db)

        rsrp_dbm = (
            cell.tx_power_dbm
            - path_loss_db
            + shadowing_db
        )

        # Penalty khi cell giảm công suất quá thấp
        if cell.tx_power_dbm <= cell.min_tx_power_dbm + 2.0:
            power_penalty_db = (
                cell.min_tx_power_dbm + 2.0 - cell.tx_power_dbm
            ) * 8.0

            rsrp_dbm -= power_penalty_db
            rsrp_dbm += rng.gauss(0.0, 3.0)

        # RSRQ measurement gần MATLAB:
        # rssi = rsrp + 10log10(12) + noise
        rssi_dbm = (
            rsrp_dbm
            + 10.0 * math.log10(12.0)
            + rng.gauss(0.0, 0.5)
        )

        rsrq_db = (
            10.0 * math.log10(12.0)
            + rsrp_dbm
            - rssi_dbm
        )
        rsrq_db = max(-20.0, min(-3.0, rsrq_db))

        # SINR measurement dB cho handover/KPI
        sinr_db = (
            rsrp_dbm
            - cfg.noise_power_dbm
            + rng.gauss(0.0, cfg.sinr_noise_std_db)
        )

        if cell.tx_power_dbm <= cell.min_tx_power_dbm + 2.0:
            sinr_penalty_db = (
                cell.min_tx_power_dbm + 2.0 - cell.tx_power_dbm
            ) * 6.0
            sinr_db -= sinr_penalty_db

        return rsrp_dbm, rsrq_db, sinr_db

    def _update_serving_measurement(self, ue: UE) -> None:
        """
        Nếu UE đã có serving cell:
        - cập nhật RSRP/RSRQ/SINR theo serving cell đó
        - nếu không còn đo được serving cell thì set NaN

        Nếu UE chưa attach:
        - không attach ở đây
        - attachment.py sẽ xử lý.
        """

        if ue.serving_cell is None:
            return

        serving_meas = None

        for meas in ue.neighbor_measurements:
            if meas.cell_id == ue.serving_cell:
                serving_meas = meas
                break

        if serving_meas is None:
            ue.rsrp = float("nan")
            ue.rsrq = float("nan")
            ue.sinr = float("nan")
            return

        ue.rsrp = serving_meas.rsrp
        ue.rsrq = serving_meas.rsrq
        ue.sinr = serving_meas.sinr
        ue.serving_ru = serving_meas.ru_id

    def _clear_radio_state(self, ue: UE) -> None:
        ue.serving_cell = None
        ue.serving_ru = None
        ue.connected = False

        ue.rsrp = float("nan")
        ue.rsrq = float("nan")
        ue.sinr = float("nan")