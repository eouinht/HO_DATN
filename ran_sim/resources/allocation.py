from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple
import math

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import UE, Cell, ResourcePool, Allocation
from radio.channel import (
    calculate_mimo_channel_gain_over_noise,
    calculate_throughput_bps,
)


# ============================================================
# Distance helper
# ============================================================

def distance_ue_cell_m(ue: UE, cell: Cell) -> float:
    return max(1.0, math.hypot(ue.x - cell.x, ue.y - cell.y))


# ============================================================
# Resource computation
# ============================================================

def compute_allocation_metrics(
    ue: UE,
    cell: Cell,
    cfg: SimulationConfig,
    rng_manager: RNGManager,
    current_time: float,
    num_prbs: int,
    tx_power_watts: float,
) -> Allocation:
    """
    Tính throughput, latency, DU/CU compute, xhaul bandwidth theo công thức env cũ.

    Công thức chính:
        gain = ||h||² / noise_power_RB
        SNR = power_per_PRB * gain
        throughput = N_PRB * B_PRB * log2(1 + SNR)

        DU cycles = k_DU * throughput
        CU cycles = k_CU * throughput
    """

    num_prbs = max(1, int(num_prbs))
    tx_power_watts = max(1e-12, float(tx_power_watts))

    distance_m = distance_ue_cell_m(ue, cell)

    rng = rng_manager.get_rng_for(
        ue.id,
        cell.id,
        int(current_time * 1000),
        7001,
    )

    gain_over_noise = calculate_mimo_channel_gain_over_noise(
        distance_m=distance_m,
        cfg=cfg,
        rng=rng,
    )

    throughput_bps = calculate_throughput_bps(
        gain_over_noise=gain_over_noise,
        tx_power_watt=tx_power_watts,
        num_prbs=num_prbs,
        prb_bandwidth_hz=cfg.prb_bandwidth_hz,
    )

    du_cycles_req = cfg.k_du_cycles_per_bit * throughput_bps
    cu_cycles_req = cfg.k_cu_cycles_per_bit * throughput_bps

    latency_s = calculate_latency_s(
        distance_m=distance_m,
        throughput_bps=throughput_bps,
        packet_size_bits=ue.packet_size_bits,
        lambda_pps=ue.lambda_pps,
        du_cycles_req=du_cycles_req,
        cu_cycles_req=cu_cycles_req,
        cfg=cfg,
    )

    return Allocation(
        ru_id=cell.ru_id,
        du_id=None,
        cu_id=None,
        num_prbs=num_prbs,
        tx_power_watts=tx_power_watts,
        throughput_bps=throughput_bps,
        latency_s=latency_s,
        du_cycles_req=du_cycles_req,
        cu_cycles_req=cu_cycles_req,
        ru_du_bw_req_bps=throughput_bps,
        du_cu_bw_req_bps=throughput_bps,
    )


def calculate_latency_s(
    distance_m: float,
    throughput_bps: float,
    packet_size_bits: int,
    lambda_pps: float,
    du_cycles_req: float,
    cu_cycles_req: float,
    cfg: SimulationConfig,
) -> float:
    """
    Latency model ổn định hơn.

    Gồm:
    - propagation delay
    - transmission delay
    - DU processing delay
    - CU processing delay
    - simple queueing delay
    - xhaul delay
    """

    eps = 1e-12
    c_speed = 3e8

    throughput_bps = max(float(throughput_bps), eps)
    packet_size_bits = max(int(packet_size_bits), 1)

    # 1. Propagation delay
    l_prop = distance_m / c_speed

    # 2. Transmission delay
    l_trans = packet_size_bits / throughput_bps

    # 3. Processing delay
    du_cycles_per_packet = cfg.k_du_cycles_per_bit * packet_size_bits
    cu_cycles_per_packet = cfg.k_cu_cycles_per_bit * packet_size_bits

    l_proc_du = du_cycles_per_packet / max(du_cycles_req, eps)
    l_proc_cu = cu_cycles_per_packet / max(cu_cycles_req, eps)

    # 4. Queueing delay, bản đơn giản và ổn định
    arrival_bits_per_s = lambda_pps * packet_size_bits
    load_ratio = arrival_bits_per_s / throughput_bps
    load_ratio = min(0.99, max(0.0, load_ratio))

    l_queue = load_ratio / max(1e-9, 1.0 - load_ratio) * l_trans

    # 5. Xhaul delay
    l_xhaul = cfg.ru_du_delay_s + cfg.du_cu_delay_s

    return float(
        l_prop
        + l_trans
        + l_proc_du
        + l_proc_cu
        + l_queue
        + l_xhaul
    )

# ============================================================
# Feasibility
# ============================================================

def check_feasible(
    allocation: Allocation,
    resource_pool: ResourcePool,
    ue: UE,
    du_id: int,
    cu_id: int,
) -> Tuple[bool, str]:
    """
    Kiểm tra allocation có thỏa resource + QoS không.
    """

    if allocation.num_prbs > resource_pool.prbs_remaining:
        return False, "insufficient PRB resource"

    ru = resource_pool.rus[allocation.ru_id - 1]
    if allocation.tx_power_watts > ru.power_remaining_watts:
        return False, "insufficient RU power resource"

    du = resource_pool.dus[du_id - 1]
    if allocation.du_cycles_req > du.remaining_cycles_per_s:
        return False, "insufficient DU compute resource"

    cu = resource_pool.cus[cu_id - 1]
    if allocation.cu_cycles_req > cu.remaining_cycles_per_s:
        return False, "insufficient CU compute resource"

    ru_du_key = (allocation.ru_id, du_id)
    du_cu_key = (du_id, cu_id)

    ru_du_link = resource_pool.ru_du_links.get(ru_du_key)
    if ru_du_link is None:
        return False, "RU-DU link not found"

    if allocation.ru_du_bw_req_bps > ru_du_link.remaining_bandwidth_bps:
        return False, "insufficient RU-DU bandwidth"

    du_cu_link = resource_pool.du_cu_links.get(du_cu_key)
    if du_cu_link is None:
        return False, "DU-CU link not found"

    if allocation.du_cu_bw_req_bps > du_cu_link.remaining_bandwidth_bps:
        return False, "insufficient DU-CU bandwidth"

    if allocation.throughput_bps < ue.min_rate_bps:
        return False, "throughput QoS violation"

    if allocation.latency_s > ue.max_latency_s:
        return False, "latency QoS violation"

    return True, "feasible"


# ============================================================
# Reserve / release resource
# ============================================================

def reserve_resources(
    allocation: Allocation,
    resource_pool: ResourcePool,
    du_id: int,
    cu_id: int,
) -> Allocation:
    """
    Trừ tài nguyên khỏi ResourcePool.
    """

    allocation.du_id = du_id
    allocation.cu_id = cu_id

    resource_pool.prbs_remaining -= allocation.num_prbs

    ru = resource_pool.rus[allocation.ru_id - 1]
    ru.power_remaining_watts -= allocation.tx_power_watts

    du = resource_pool.dus[du_id - 1]
    du.remaining_cycles_per_s -= allocation.du_cycles_req

    cu = resource_pool.cus[cu_id - 1]
    cu.remaining_cycles_per_s -= allocation.cu_cycles_req

    resource_pool.ru_du_links[(allocation.ru_id, du_id)].remaining_bandwidth_bps -= allocation.ru_du_bw_req_bps
    resource_pool.du_cu_links[(du_id, cu_id)].remaining_bandwidth_bps -= allocation.du_cu_bw_req_bps

    return allocation


def release_resources(
    allocation: Allocation,
    resource_pool: ResourcePool,
) -> None:
    """
    Hoàn trả tài nguyên allocation cũ.
    """

    if allocation.ru_id is None or allocation.du_id is None or allocation.cu_id is None:
        return

    resource_pool.prbs_remaining += allocation.num_prbs

    ru = resource_pool.rus[allocation.ru_id - 1]
    ru.power_remaining_watts += allocation.tx_power_watts

    du = resource_pool.dus[allocation.du_id - 1]
    du.remaining_cycles_per_s += allocation.du_cycles_req

    cu = resource_pool.cus[allocation.cu_id - 1]
    cu.remaining_cycles_per_s += allocation.cu_cycles_req

    ru_du_key = (allocation.ru_id, allocation.du_id)
    du_cu_key = (allocation.du_id, allocation.cu_id)

    if ru_du_key in resource_pool.ru_du_links:
        resource_pool.ru_du_links[ru_du_key].remaining_bandwidth_bps += allocation.ru_du_bw_req_bps

    if du_cu_key in resource_pool.du_cu_links:
        resource_pool.du_cu_links[du_cu_key].remaining_bandwidth_bps += allocation.du_cu_bw_req_bps