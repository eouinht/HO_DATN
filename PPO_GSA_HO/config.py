import numpy as np

DEBUG_RADIO_METRICS = True
DEBUG_RADIO_UE_ID = None
RADIO_LOG_DIR = "./logs"

# =======================================================
# ================== Tham số mô phỏng ===================
# =======================================================
# ---------------------- Quy mô mạng -----------------------
# ---------------------- RB & băng thông -------------------
# NR: SCS=30kHz, 1 RB = 12 subcarriers => 360 kHz
subcarrier_bandwidth_Hz = 30e3
num_subcarriers_per_RB  = 12
channel_bandwidth_Hz    = 100e6

#bandwidth_per_RB        = channel_bandwidth_Hz / num_RBs
bandwidth_per_RB        = num_subcarriers_per_RB * subcarrier_bandwidth_Hz


MAX_RBS_PER_SLICE = {
    "eMBB": 20,
    "uRLLC": 10,
}
max_RBs_per_UE = max(MAX_RBS_PER_SLICE.values())
# ---------------------- Cấu hình dịch vụ / slice ----------------------
SLICE_PRESET = {
    'eMBB': {
        'type'        : 'eMBB',
        'R_min'       : 50e6,        # [bps]
        'SINR_min'    : 10,          # [dB]
        'eta_slice'   : 0.05,
        'weight_accept'    : 10.0,
        'weight_throughput'     : 2.0,
        'weight_latency'     : 1.0,
        'weight_handover'     : 10.0,
        'delay'             : 5e-3,         # [s]
        'packet_size_bits'  : 1500 * 8,
        'cycles_per_packet' : 4000.0,
        'lambda_default_pps': 100.0,
        'max_RBs': MAX_RBS_PER_SLICE["eMBB"],
        
    },
    'uRLLC': {
        'type'        : 'uRLLC',
        'R_min'       : 5e6,
        'SINR_min'    : 20,          # [dB]
        'eta_slice'   : 0.08,
        'weight_accept'    : 10.0,
        'weight_throughput'     : 2.0,
        'weight_latency'     : 1.0,
        'weight_handover'     : 10.0,
        'delay'             : 1e-3,        # [s]
        'packet_size_bits'  : 128 * 8,
        'cycles_per_packet' : 2000.0,
        'lambda_default_pps': 500.0,
        'max_RBs': MAX_RBS_PER_SLICE["uRLLC"],
    },
}

# ---------------------- Công suất --------------------
# RU: công suất phát tối đa ~ 43 dBm ≈ 20 W
max_tx_power_dbm    = 43
max_tx_power_mw     = 10 ** (max_tx_power_dbm / 10)  # mW
max_tx_power_watts  = max_tx_power_mw / 1e3          # W

# ---------------------- Tài nguyên node mạng -----------------
P_i_random_list = [max_tx_power_watts]              # RU powers (W)
A_j_random_list = [8e9]                             # DU CPU (cycles/s)
A_m_random_list = [5e9]                             # CU CPU (cycles/s)
# -------------- Tài nguyên liên kết trong mạng ---------------
bw_ru_du_random_list = [5e9]                               # bps
bw_du_cu_random_list = [20e9]                              # bps

# ---------------------- Mức công suất rời rạc cho agent (RU) --------------
num_power_levels = 10
def generate_power_levels(P_max, num_power_levels):
    if num_power_levels <= 1:
        return [float(P_max)]
    arr = np.linspace(P_max / num_power_levels, P_max, num_power_levels, dtype=float)
    return [float(round(x, 6)) for x in arr]

P_ib_sk_val = generate_power_levels(max_tx_power_watts, num_power_levels)

# ---------------------- Mô hình CPU theo bit ----------------
k_DU = 5.0     # [cycles/bit] tại DU
k_CU = 3.0      # [cycles/bit] tại CU

import numpy as np

# =======================================================
# UE POPULATION
# =======================================================
UE_POPULATION_MARGIN = 5
BASE_ARRIVAL_LAMBDA = 0.8
POPULATION_CORRECTION_GAIN = 0.25
MAX_UE_ARRIVALS_PER_STEP = 3
MAX_RECOVERY_ARRIVALS_PER_STEP = 8

# =======================================================
# UE MOBILITY - DISCRETE STEP MODEL
# =======================================================
UE_STEP_DISTANCE_MIN = 20.0
UE_STEP_DISTANCE_MAX = 60.0
UE_STEP_DISTANCE_JITTER = 5.0
UE_DIRECTION_JITTER_RAD = np.deg2rad(15.0)

# =======================================================
# UE SESSION LIFECYCLE
# =======================================================
MIN_SESSION_STEPS = 10
MEAN_EXTRA_SESSION_STEPS = 20
MAX_SESSION_STEPS = 80

# =======================================================
# QUALITY-BASED DROP
# =======================================================
REFERENCE_POWER_PER_RB_W = 1.0
SNR_DROP_THRESHOLD_DB = 5.0
DROP_TTT_STEPS = 3
NEW_UE_DROP_GRACE_STEPS = 3

# =======================================================
# UE SPAWN
# =======================================================
BOUNDARY_SPAWN_RATIO = 0.7
BOUNDARY_SPAWN_MIN_RATIO = 0.75
BOUNDARY_SPAWN_MAX_RATIO = 1.0

SLICE_PROBABILITIES = {
    "eMBB": 0.7,
    "uRLLC": 0.3,
}