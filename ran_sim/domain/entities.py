from dataclasses import dataclass, field
from typing import Optional
 
@dataclass
class Site:
    id: int
    x: float
    y: float
    site_type: str

@dataclass 
class RUNode:
    id: int
    site_id: int
    x: float
    y: float
    power_capacity_watts: float
    power_remaining_watts: float

@dataclass
class DUNode:
    id: int
    capacity_cycles_per_s: float
    remaining_cycles_per_s: float
    
@dataclass
class CUNode:
    id: int 
    capacity_cycles_per_s: float
    remaining_cycles_per_s: float

@dataclass
class TransportLink:
    src_id: int
    dst_id: int
    bandwidth_bps: float
    remaining_bandwidth_bps: float
    delay_s: float
    link_type: str  # "RU-DU" or "DU-CU"


@dataclass
class ResourcePool:
    total_prbs: int
    prbs_remaining: int

    rus: list[RUNode] = field(default_factory=list)
    dus: list[DUNode] = field(default_factory=list)
    cus: list[CUNode] = field(default_factory=list)

    ru_du_links: dict[tuple[int, int], TransportLink] = field(default_factory=dict)
    du_cu_links: dict[tuple[int, int], TransportLink] = field(default_factory=dict)


@dataclass
class NeighborMeasurement:
    cell_id: int
    ru_id: int
    rsrp: float
    rsrq: float
    sinr: float


@dataclass
class Allocation:
    ru_id: Optional[int] = None
    du_id: Optional[int] = None
    cu_id: Optional[int] = None

    num_prbs: int = 0
    tx_power_watts: float = 0.0

    throughput_bps: float = 0.0
    latency_s: float = 0.0

    du_cycles_req: float = 0.0
    cu_cycles_req: float = 0.0

    ru_du_bw_req_bps: float = 0.0
    du_cu_bw_req_bps: float = 0.0


@dataclass
class UE:
    id: int
    x: float
    y: float

    velocity_mps: float
    direction_rad: float
    mobility_pattern: str

    slice_type: str = "eMBB"

    serving_cell: Optional[int] = None
    serving_ru: Optional[int] = None

    rsrp: float = float("nan")
    rsrq: float = float("nan")
    sinr: float = float("nan")

    neighbor_measurements: list[NeighborMeasurement] = field(default_factory=list)

    traffic_demand_bps: float = 0.0
    packet_size_bits: int = 0
    lambda_pps: float = 0.0
    session_active: bool = False

    min_rate_bps: float = 0.0
    max_latency_s: float = 0.0
    min_sinr_db: float = 0.0

    allocation: Allocation = field(default_factory=Allocation)
    prev_allocation: Allocation = field(default_factory=Allocation)

    handover_count: int = 0
    pingpong_count: int = 0
    drop_count: int = 0

    ho_timer_s: float = 0.0
    connection_timer_s: float = 0.0
    disconnection_timer_s: float = 0.0

    train_start_x_m: float = 0.0
    track_length_m: float = 0.0
    position_in_train_m: float = 0.0

    step_counter: int = 0
    pause_timer_s: float = 0.0
    last_direction_change_s: float = 0.0

    active: bool = True
    connected: bool = False


@dataclass
class Cell:
    id: int
    ru_id: int
    site_id: int
    sector_id: int

    x: float
    y: float
    azimuth_deg: float

    frequency_hz: float
    antenna_height_m: float
    cell_radius_m: float

    tx_power_dbm: float
    min_tx_power_dbm: float
    max_tx_power_dbm: float

    a3_offset_db: float
    ttt_ms: float

    connected_ues: list[int] = field(default_factory=list)

    current_load_bps: float = 0.0
    prb_used: int = 0
    avg_rsrp: float = 0.0
    avg_rsrq: float = 0.0
    avg_sinr: float = 0.0

    energy_consumption_watt: float = 0.0
    drop_rate_percent: float = 0.0
    avg_latency_ms: float = 0.0


@dataclass
class HandoverEvent:
    ue_id: int
    cell_source: int
    cell_target: int

    ru_source: Optional[int]
    ru_target: Optional[int]

    rsrp_source: float
    rsrp_target: float
    rsrq_source: float
    rsrq_target: float
    sinr_source: float
    sinr_target: float

    a3_offset_db: float
    ttt_ms: float

    ho_success: bool
    timestamp_s: float


@dataclass
class EnergyMetrics:
    time_s: float

    total_energy_kwh: float
    instantaneous_power_watt: float

    connected_ues: int
    connection_rate_percent: float

    total_traffic_bps: float
    avg_drop_rate_percent: float
    avg_latency_ms: float

    total_handovers: int
    handover_success_rate: float

    prb_usage_percent: float
    ru_power_usage_percent: float
    du_usage_percent: float
    cu_usage_percent: float
    xhaul_usage_percent: float

    prb_violations: int
    ru_power_violations: int
    du_violations: int
    cu_violations: int
    xhaul_violations: int


@dataclass
class CellStateSnapshot:
    time_s: float
    cell_states: list[dict]
    resource_state: dict
    cumulative_energy_kwh: float


@dataclass
class UEStateSnapshot:
    time_s: float
    ue_states: list[dict]


@dataclass
class SimulationResults:
    handover_events: list[HandoverEvent] = field(default_factory=list)
    energy_metrics: list[EnergyMetrics] = field(default_factory=list)
    cell_states: list[CellStateSnapshot] = field(default_factory=list)
    ue_trajectories: list[UEStateSnapshot] = field(default_factory=list)

    total_simulation_time_s: float = 0.0
    total_handovers: int = 0
    successful_handovers: int = 0
    final_success_rate: float = 0.0


@dataclass
class SimulationSnapshot:
    time_s: float
    step_idx: int

    sites: list[Site]
    cells: list[Cell]
    ues: list[UE]
    resource_pool: ResourcePool

    results: SimulationResults
    done: bool