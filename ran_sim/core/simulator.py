from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class NeighborMeasurement:
    cell_id: int
    rsrp: float
    rsrq: float
    sinr: float

@dataclass
class UE:
    id: int
    x: float
    y: float
    velocity: float
    direction: float
    mobility_pattern: str

    serving_cell: Optional[int] = None
    rsrp: float = float("nan")
    rsrq: float = float("nan")
    sinr: float = float("nan")
    neighbor_measurements: List[NeighborMeasurement] = field(default_factory=list)

    traffic_demand: float = 0.0
    session_active: bool = False
    drop_count: int = 0

    ho_timer: float = 0.0
    connection_timer: float = 0.0
    disconnection_timer: float = 0.0

    step_counter: int = 0
    last_direction_change: float = 0.0
    pause_timer: float = 0.0

@dataclass
class Cell:
    id: int
    site_id: int
    sector_id: int
    x: float
    y: float
    frequency: float
    tx_power: float
    min_tx_power: float
    max_tx_power: float
    cell_radius: float
    base_energy_consumption: float
    idle_energy_consumption: float
    max_capacity: float
    ttt: float
    a3_offset: float

    cpu_usage: float = 0.0
    prb_usage: float = 0.0
    energy_consumption: float = 0.0
    current_load: float = 0.0
    connected_ues: List[int] = field(default_factory=list)
    avg_sinr: float = 0.0
    drop_rate: float = 0.0
    avg_latency: float = 0.0