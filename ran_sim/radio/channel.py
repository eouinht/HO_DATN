import math
import numpy as np

from core.config import SimulationConfig
from radio.pathloss import calculate_path_loss_db


def thermal_noise_power_watt(
    bandwidth_hz: float,
    noise_figure_db: float = 5.0,
    temperature_k: float = 290.0,
) -> float:
    """
    Noise power:
        N = k_B * T * B * NF

    Đây là công thức env cũ đang dùng.
    """

    k_b = 1.38064852e-23
    n0_w_per_hz = k_b * temperature_k
    noise_figure_linear = 10.0 ** (noise_figure_db / 10.0)

    return n0_w_per_hz * bandwidth_hz * noise_figure_linear


def calculate_mimo_channel_gain_over_noise(
    distance_m: float,
    cfg: SimulationConfig,
    rng,
    num_antennas: int = 32,
) -> float:
    """
    Tính gain / noise_power_RB giống env cũ.

    Env cũ:
        path_loss_db = 28 + 20log10(fc_GHz) + 22log10(d)
        path_loss_linear = 10 ** (-path_loss_db / 10)

        h = sqrt(path_loss_linear) * (h_real + j h_imag) / sqrt(2)
        channel = ||h||_2^2
        gain = channel / noise_power_RB

    Return:
        gain_over_noise
    """

    distance_m = max(float(distance_m), 1.0)

    path_loss_db = calculate_path_loss_db(
        distance_m=distance_m,
        carrier_frequency_hz=cfg.carrier_frequency,
    )

    path_loss_linear = 10.0 ** (-path_loss_db / 10.0)

    h_real = np.array([rng.gauss(0.0, 1.0) for _ in range(num_antennas)])
    h_imag = np.array([rng.gauss(0.0, 1.0) for _ in range(num_antennas)])

    h = math.sqrt(path_loss_linear) * (h_real + 1j * h_imag) / math.sqrt(2.0)

    channel_power_gain = np.linalg.norm(h, 2) ** 2

    noise_power_rb = thermal_noise_power_watt(
        bandwidth_hz=cfg.prb_bandwidth_hz,
        noise_figure_db=5.0,
        temperature_k=290.0,
    )

    return float(channel_power_gain / max(noise_power_rb, 1e-30))


def calculate_snr_linear(
    gain_over_noise: float,
    tx_power_watt: float,
    num_prbs: int,
) -> float:
    """
    SNR theo công thức env cũ:

        power_per_RB = power_alloc / num_RB_alloc
        SNR_per_RB = power_per_RB * gain
    """

    num_prbs = max(int(num_prbs), 1)
    power_per_prb = float(tx_power_watt) / num_prbs

    return max(0.0, power_per_prb * float(gain_over_noise))


def calculate_throughput_bps(
    gain_over_noise: float,
    tx_power_watt: float,
    num_prbs: int,
    prb_bandwidth_hz: float,
) -> float:
    """
    Throughput theo env cũ:

        R = N_RB * B_RB * log2(1 + SNR)
    """

    snr = calculate_snr_linear(
        gain_over_noise=gain_over_noise,
        tx_power_watt=tx_power_watt,
        num_prbs=num_prbs,
    )

    return float(num_prbs * prb_bandwidth_hz * math.log2(1.0 + snr))


def watt_to_dbm(power_watt: float) -> float:
    power_watt = max(float(power_watt), 1e-30)
    return 10.0 * math.log10(power_watt * 1000.0)


def dbm_to_watt(power_dbm: float) -> float:
    return 10.0 ** ((float(power_dbm) - 30.0) / 10.0)