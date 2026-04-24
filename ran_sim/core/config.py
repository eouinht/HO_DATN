from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # =====================================================
    # Simulation time
    # =====================================================
    sim_time: float = 300.0
    time_step: float = 1.0

    # =====================================================
    # Deployment scenario
    # =====================================================
    deployment_scenario: str = "dense_urban"
    # indoor_hotspot | dense_urban | rural | urban_macro
    # high_speed | extreme_rural

    # =====================================================
    # Network topology: 7 RU - 3 DU - 3 CU
    # =====================================================
    num_sites: int = 7
    num_rus: int = 7
    num_dus: int = 3
    num_cus: int = 3
    num_ues: int = 50

    num_sectors: int = 3
    isd: float = 200.0
    max_radius: float = 500.0

    # =====================================================
    # Dynamic UE lifecycle
    # =====================================================
    enable_dynamic_ues: bool = True

    min_ues: int = 10
    max_ues: int = 100

    max_ue_arrivals_per_step: int = 2
    max_ue_departures_per_step: int = 2

    ue_arrival_probability: float = 0.5
    ue_departure_probability: float = 0.3

    # =====================================================
    # Radio / channel parameters
    # =====================================================
    carrier_frequency: float = 2.6e9     # Hz = 2.6 GHz
    antenna_height: float = 25.0
    cell_radius: float = 200.0

    rsrp_measurement_threshold: float = -120.0
    rsrp_serving_threshold: float = -110.0
    rsrp_target_threshold: float = -105.0

    noise_power_dbm: float = -110.0
    shadowing_std_db: float = 1.5
    sinr_noise_std_db: float = 2.0

    # =====================================================
    # Handover parameters
    # =====================================================
    default_a3_offset: float = 8.0
    default_ttt_ms: float = 8.0
    hysteresis_margin: float = 3.0

    disconnection_timeout: float = 5.0
    connection_timeout: float = 2.0

    # =====================================================
    # Shared PRB resource pool
    # =====================================================
    # 100 MHz, SCS = 30 kHz thường tương ứng khoảng 273 PRBs.

    total_prbs: int = 273
    prb_bandwidth_hz: float = 360e3
    max_prbs_per_ue: int = 10

    # =====================================================
    # Shared RU power resource pool
    # Giống env cũ: mỗi RU lấy capacity từ choices
    # ResourcePool sẽ sinh RU_power_capacity shape = (num_rus,)
    # =====================================================
    ru_power_capacity_choices_watts: tuple[float] = (20.0,)

    min_tx_power_dbm: float = 30.0
    max_tx_power_dbm: float = 46.0
    initial_tx_power_dbm: float = 43.0

    # =====================================================
    # Shared DU / CU compute resource pools
    # Giống env cũ:
    #   DU_capacity shape = (num_dus,)
    #   CU_capacity shape = (num_cus,)
    # =====================================================
    du_capacity_choices_cycles_per_s: tuple[float] = (8e9,)
    cu_capacity_choices_cycles_per_s: tuple[float] = (5e9,)

    k_du_cycles_per_bit: float = 5.0
    k_cu_cycles_per_bit: float = 3.0

    # =====================================================
    # Shared transport / xhaul resource pools
    # RU-DU bandwidth matrix shape = (num_rus, num_dus)
    # DU-CU bandwidth matrix shape = (num_dus, num_cus)
    # =====================================================
    ru_du_bandwidth_choices_bps: tuple[float] = (5e9,)
    du_cu_bandwidth_choices_bps: tuple[float] = (20e9,)

    ru_du_delay_s: float = 1e-3
    du_cu_delay_s: float = 2e-3

    # =====================================================
    # Energy model
    # =====================================================
    base_power_watt: float = 800.0
    idle_power_watt: float = 200.0
    per_ue_power_watt: float = 15.0
    load_power_watt: float = 200.0

    # =====================================================
    # Traffic model
    # =====================================================
    traffic_lambda: float = 10.0
    peak_hour_multiplier: float = 1.0

    embb_ratio: float = 0.7
    urllc_ratio: float = 0.3

    embb_packet_size_bits: int = 1500 * 8
    urllc_packet_size_bits: int = 128 * 8

    embb_lambda_pps: float = 100.0
    urllc_lambda_pps: float = 500.0

    # =====================================================
    # Slice QoS requirements
    # =====================================================
    embb_min_rate_bps: float = 50e6
    embb_max_latency_s: float = 5e-3
    embb_min_sinr_db: float = 10.0

    urllc_min_rate_bps: float = 5e6
    urllc_max_latency_s: float = 1e-3
    urllc_min_sinr_db: float = 20.0

    # =====================================================
    # KPI thresholds
    # =====================================================
    drop_call_threshold_percent: float = 5.0
    latency_threshold_ms: float = 5.0

    prb_usage_threshold_percent: float = 90.0
    ru_power_usage_threshold_percent: float = 90.0
    du_usage_threshold_percent: float = 90.0
    cu_usage_threshold_percent: float = 90.0
    xhaul_usage_threshold_percent: float = 90.0

    # =====================================================
    # Mobility parameters
    # =====================================================
    ue_speed_kmh: float = 30.0
    indoor_ratio: float = 0.8

    train_length_m: float = 200.0
    track_length_m: float = 10000.0

    indoor_width_m: float = 120.0
    indoor_height_m: float = 50.0

    # =====================================================
    # Random seed
    # =====================================================
    seed: int = 42

    # =====================================================
    # Logging
    # =====================================================
    log_dir: str = "logs"
    log_file: str = "logs/sim.log"
    ue_log_file: str = "logs/ue.log"
    handover_log_file: str = "logs/handover.log"
    agent_log_file: str = "logs/agent.log"

    # =====================================================
    # Derived properties
    # =====================================================
    @property
    def num_steps(self) -> int:
        return round(self.sim_time / self.time_step)

    @property
    def carrier_frequency_ghz(self) -> float:
        return self.carrier_frequency / 1e9

    @property
    def ue_speed_mps(self) -> float:
        return self.ue_speed_kmh / 3.6

    @property
    def default_ru_power_capacity_watts(self) -> float:
        return self.ru_power_capacity_choices_watts[0]

    @property
    def default_du_capacity_cycles_per_s(self) -> float:
        return self.du_capacity_choices_cycles_per_s[0]

    @property
    def default_cu_capacity_cycles_per_s(self) -> float:
        return self.cu_capacity_choices_cycles_per_s[0]

    @property
    def default_ru_du_bandwidth_bps(self) -> float:
        return self.ru_du_bandwidth_choices_bps[0]

    @property
    def default_du_cu_bandwidth_bps(self) -> float:
        return self.du_cu_bandwidth_choices_bps[0]

    @property
    def total_ru_power_capacity_watts(self) -> float:
        return self.num_rus * self.default_ru_power_capacity_watts

    @property
    def total_du_capacity_cycles_per_s(self) -> float:
        return self.num_dus * self.default_du_capacity_cycles_per_s

    @property
    def total_cu_capacity_cycles_per_s(self) -> float:
        return self.num_cus * self.default_cu_capacity_cycles_per_s

    @property
    def total_ru_du_bandwidth_bps(self) -> float:
        return self.num_rus * self.num_dus * self.default_ru_du_bandwidth_bps

    @property
    def total_du_cu_bandwidth_bps(self) -> float:
        return self.num_dus * self.num_cus * self.default_du_cu_bandwidth_bps

    @property
    def initial_power_ratio(self) -> float:
        return (
            self.initial_tx_power_dbm - self.min_tx_power_dbm
        ) / max(1e-9, self.max_tx_power_dbm - self.min_tx_power_dbm)