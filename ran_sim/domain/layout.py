from typing import List
import math

from core.config import SimulationConfig
from core.rng import RNGManager
from domain.entities import Site

import matplotlib.pyplot as plt

# ============================================================
# Public API
# ============================================================

def create_layout(cfg: SimulationConfig, rng: RNGManager) -> List[Site]:
    """
    Sinh danh sách Site theo deployment_scenario.
    """
    creators = {
        "dense_urban": _create_hex_layout,
        "urban_macro": _create_hex_layout,
        "rural": _create_hex_layout,
        "extreme_rural": _create_hex_layout,
        "high_speed": _create_high_speed_layout,
        "indoor_hotspot": _create_indoor_layout,
    }
    creator = creators.get(cfg.deployment_scenario, _create_hex_layout)
    sites = creator(cfg, rng)
    return sites


# ============================================================
# Hex / macro layouts (7 sites mặc định)
# ============================================================

def _create_hex_layout(cfg: SimulationConfig, rng: RNGManager) -> List[Site]:
    """
    1 site trung tâm + các site xung quanh trên vòng tròn (gần hex grid).
    Dùng cho dense_urban / urban_macro / rural / extreme_rural.
    """
    sites: List[Site] = []

    # Site trung tâm
    sites.append(Site(id=1, x=0.0, y=0.0, site_type="macro"))

    if cfg.num_sites == 1:
        return sites

    # Các site còn lại trên vòng tròn bán kính ISD
    num_outer = cfg.num_sites - 1
    angle_step = 2 * math.pi / num_outer
    radius = cfg.isd

    for i in range(num_outer):
        angle = i * angle_step

        # jitter nhỏ để tránh đối xứng hoàn toàn
        jitter_r = radius * 0.05
        jitter = rng.get_rng_for(i).uniform(-jitter_r, jitter_r)

        r = radius + jitter

        x = r * math.cos(angle)
        y = r * math.sin(angle)

        sites.append(
            Site(
                id=i + 2,
                x=x,
                y=y,
                site_type="macro",
            )
        )

    return sites


# ============================================================
# High-speed layout (dọc tuyến đường)
# ============================================================

def _create_high_speed_layout(cfg: SimulationConfig, rng: RNGManager) -> List[Site]:
    """
    Các site đặt dọc theo trục x (giống tuyến đường sắt cao tốc).
    """
    sites: List[Site] = []

    total_length = (cfg.num_sites - 1) * cfg.isd
    start_x = -total_length / 2.0

    for i in range(cfg.num_sites):
        x = start_x + i * cfg.isd

        # jitter nhỏ theo trục y
        jitter_y = rng.get_rng_for(i).uniform(-10.0, 10.0)

        sites.append(
            Site(
                id=i + 1,
                x=x,
                y=100.0 + jitter_y,
                site_type="high_speed_rrh",
            )
        )

    return sites


# ============================================================
# Indoor layout (grid trong nhà)
# ============================================================

def _create_indoor_layout(cfg: SimulationConfig, rng: RNGManager) -> List[Site]:
    """
    Grid nhỏ trong không gian indoor.
    """
    sites: List[Site] = []

    cols = max(1, int(math.sqrt(cfg.num_sites)))
    rows = math.ceil(cfg.num_sites / cols)

    dx = cfg.indoor_width_m / (cols + 1)
    dy = cfg.indoor_height_m / (rows + 1)

    idx = 1
    for r in range(rows):
        for c in range(cols):
            if idx > cfg.num_sites:
                break

            # jitter nhỏ
            jitter_x = rng.get_rng_for(idx, 1).uniform(-1.0, 1.0)
            jitter_y = rng.get_rng_for(idx, 2).uniform(-1.0, 1.0)

            x = (c + 1) * dx + jitter_x
            y = (r + 1) * dy + jitter_y

            sites.append(
                Site(
                    id=idx,
                    x=x,
                    y=y,
                    site_type="indoor_trx",
                )
            )

            idx += 1

    return sites

# ============================================================
# Plot layout
# ============================================================

def plot_layout(
    sites: List[Site],
    cfg: SimulationConfig,
    save_path: str | None = None,
    show: bool = True,
):
    """
    Vẽ layout mạng để đưa vào báo cáo / đồ án.

    Parameters
    ----------
    sites:
        Danh sách Site đã tạo bởi create_layout().
    cfg:
        SimulationConfig.
    save_path:
        Đường dẫn lưu hình. Ví dụ: "figures/layout_dense_urban.png"
    show:
        True nếu muốn hiển thị hình.
    """
    

    fig, ax = plt.subplots(figsize=(7, 6))

    xs = [s.x for s in sites]
    ys = [s.y for s in sites]

    ax.scatter(xs, ys, s=120, marker="^", label="RU / Site")

    for site in sites:
        ax.text(
            site.x,
            site.y,
            f"RU{site.id}",
            fontsize=10,
            ha="center",
            va="bottom",
        )

        # Vẽ vùng phủ sóng gần đúng
        coverage = plt.Circle(
            (site.x, site.y),
            cfg.cell_radius,
            fill=False,
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
        ax.add_patch(coverage)

    ax.set_title(
        f"Network Layout: {cfg.deployment_scenario} "
        f"({cfg.num_rus} RU - {cfg.num_dus} DU - {cfg.num_cus} CU)"
    )
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axis("equal")
    ax.legend()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax