import math


def calculate_path_loss_db(
    distance_m: float,
    carrier_frequency_hz: float,
) -> float:
    """
    Path loss model đơn giản, gần với mô hình cũ:
    PL(dB) = 28 + 20log10(fc_GHz) + 22log10(d)

    Parameters
    ----------
    distance_m:
        Khoảng cách UE-RU/cell, đơn vị mét.
    carrier_frequency_hz:
        Tần số sóng mang, đơn vị Hz.

    Returns
    -------
    path_loss_db:
        Suy hao đường truyền, đơn vị dB.
    """

    distance_m = max(distance_m, 1.0)
    carrier_frequency_ghz = carrier_frequency_hz / 1e9

    path_loss_db = (
        28.0
        + 20.0 * math.log10(carrier_frequency_ghz)
        + 22.0 * math.log10(distance_m)
    )

    return path_loss_db

def db_to_linear(db_value: float) -> float:
    return 10.0 ** (db_value / 10.0)

def linear_to_db(linear_value: float) -> float:
    linear_value = max(float(linear_value), 1e-30)
    return 10.0 * math.log10(linear_value)