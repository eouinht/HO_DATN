import numpy as np, copy
# Local modules - must exist
from numpy.linalg import norm
import networkx as nx
from datetime import datetime
from config import *
import os
import csv
from pathlib import Path
from ue import UEManager

# class UEManager:
#     def __init__(self, coordinates_RU,target_num_UEs=50):
#         self.radius_in = 10
#         self.radius_out = 1000
#         self.SLICE_PRESET = dict(SLICE_PRESET)
#         self.slice_names = list(self.SLICE_PRESET.keys())
#         self.UE_requests = {}
#         self.coordinates_RU = coordinates_RU
#         self.target_num_UEs = target_num_UEs

#         margin = max(1, int(round(self.target_num_UEs*UE_POPULATION_MARGIN_RATIO)))
#         self.min_num_UEs = max(1, self.target_num_UEs - margin)
#         self.max_num_UEs = self.target_num_UEs + margin
        
#     def initialize_mobility_state(self):
#         """
#         Khởi tạo trạng thái mobility cho một UE.

#         Mỗi UE có:
#             - speed_mps: tốc độ hiện tại [m/s]
#             - direction_rad: hướng di chuyển [rad]
#         """
#         return {
#             "speed_mps": float(
#                 np.random.uniform(
#                     UE_SPEED_MIN_MPS,
#                     UE_SPEED_MAX_MPS,
#                 )
#             ),

#             "direction_rad": float(
#                 np.random.uniform(
#                     0.0,
#                     2.0 * np.pi,
#                 )
#             ),
#         }
        
#     def empty_status(self):
#         return {"active": 1, "served": False, "reason": None}
    
#     def empty_alloc(self):
#         return {
#             "RU": None,
#             "DU": None,
#             "CU": None,
#             "num_RB_alloc": 0,
#             "power_alloc": 0.0,
#             "throughput_bps": 0.0,
#             "delay_s": 0.0,
#             "cpu_DU_req": 0.0,
#             "cpu_CU_req": 0.0,
#         }

#     def calculate_distances(self, coordinate_UE):
#         distances_RU_UE = []
#         x_UE, y_UE = coordinate_UE
#         for (x_RU, y_RU) in self.coordinates_RU:
#             d = np.sqrt((x_RU - x_UE)**2 + (y_RU - y_UE)**2)
#             distances_RU_UE.append(d)
#         return distances_RU_UE


#     def calculate_gain(self, distances_RU_UE):
#         """
#         distances_RU_UE: ma trận [num_RUs x num_UEs] khoảng cách RU-UE (m)
#         bandwidth_per_RB: băng thông 1 RB (Hz)
#         """
#         # ------------------- Antenna config -------------------
#         num_antennas = 32  # anten mỗi RU
        
#         # ------------------- Noise power ----------------------
#         k_B = 1.38064852e-23   # Boltzmann constant (J/K)
#         T_K = 290              # Nhiệt độ (K)
#         N0_W_per_Hz = k_B * T_K
#         noise_figure_dB = 5
#         noise_figure_linear = 10 ** (noise_figure_dB / 10)
#         noise_power_RB = N0_W_per_Hz * bandwidth_per_RB * noise_figure_linear
        
#         # ------------------- Carrier frequency ----------------
#         f_c_GHz = 6
#         gain = []
#         for d in distances_RU_UE:
#             # ------------------- Pathloss model (3GPP UMa) --------
#             path_loss_db = 28 + 20 * np.log10(f_c_GHz) + 22 * np.log10(d)

#             # ------------------- Pathloss linear ------------------
#             path_loss_linear = 10 ** (-path_loss_db / 10)


#             # kênh MIMO Rayleigh (num_antennas anten)
#             h_real = np.random.randn(num_antennas)
#             h_imag = np.random.randn(num_antennas)
#             h = np.sqrt(path_loss_linear) * (h_real + 1j*h_imag) / np.sqrt(2)
                
#             # power gain (chuẩn hóa theo norm-2)
#             channel = norm(h, 2) ** 2

#             gain.append(channel / noise_power_RB)
#         return gain

#     def calculate_best_reference_snr_db(self,ue_info):
#         """
#         Tính SNR tham chiếu tốt nhất mà UE có thể nhận được
#         từ toàn bộ RU.

#         gain trong mô hình đã được chuẩn hóa theo noise power
#         của một RB:

#             gain = channel_gain / noise_power_RB

#         Do đó:

#             SNR_ref = reference_power_per_RB * gain
#         """
#         gains = np.asarray(ue_info.get("gain",[],),dtype=float,)
        
#         if gains.size == 0:
#             return -float("inf")

#         snr_linear = (
#             REFERENCE_POWER_PER_RB_W
#             *
#             gains
#         )

#         best_snr_linear = float(
#             np.max(
#                 snr_linear
#             )
#         )

#         best_snr_db = 10.0 * np.log10(
#             max(
#                 best_snr_linear,
#                 1e-30,
#             )
#         )

#         return float(
#             best_snr_db
#         )
    
#     def compute_UE_metrics(self, coordinate_UE):
#         distances = self.calculate_distances(coordinate_UE)
#         gain = self.calculate_gain(distances)
#         return distances, gain

#     def set_coordinate_UE(self):
#         angles = np.random.uniform(0, 2 * np.pi)
#         r = np.random.uniform(self.radius_in, self.radius_out)

#         x = r * np.cos(angles)
#         y = r * np.sin(angles)

#         new_coords = (x, y)

#         return new_coords
    
#     def is_valid_position(self, coord):
#         x, y = coord
#         dist = np.hypot(x, y)
#         return self.radius_in <= dist <= self.radius_out


#     def build_UE_request(self, ue_id):
#         slice_name = np.random.choice(self.slice_names, p=[0.7, 0.3])
#         coordinate_UE = self.set_coordinate_UE()
#         distances_RU_UE = self.calculate_distances(coordinate_UE)
#         gains_UE = self.calculate_gain(distances_RU_UE)

#         self.UE_requests[ue_id] = {
#             "id": int(ue_id),
#             **copy.deepcopy(self.SLICE_PRESET[slice_name]),

#             "coordinate": coordinate_UE,
#             "distances_RU_UE": distances_RU_UE,
#             "gain": gains_UE,

#             "status": self.empty_status(),

#             "allocation": self.empty_alloc(),
#             "prev_allocation": self.empty_alloc(),
            
#             "pingpong": 0,
#             "handover_count": 0,
#             "mobility": self.initialize_mobility_state(),
#             "coverage": {
#                 "drop_counter": 0,
#                 "best_reference_snr_db": None,
#             },
#         }
    
    
#     def add_UEs_requests(self, new_UEs_request):
#         if new_UEs_request <= 0:
#             return []
#         start_id = max(self.UE_requests.keys(), default=-1) + 1

#         new_ids = list(range(start_id, start_id + new_UEs_request))

#         for ue_id in new_ids:
#             self.build_UE_request(ue_id)

#         return new_ids

#     def remove_UE(self, ue_id):
#         return self.UE_requests.pop(ue_id, None)

#     def remove_random_UEs(self, n_UEs_remove):
#         ue_ids = list(self.UE_requests.keys())

#         n_UEs_remove = min(int(n_UEs_remove), len(ue_ids))

#         remove_ids = list(np.random.choice(ue_ids, n_UEs_remove, replace=False))

#         removed_ues = []
#         for ue_id in remove_ids:
#             ue_info = self.remove_UE(ue_id)
#             removed_ues.append((ue_id, ue_info))

#         return removed_ues

    
#     def UE_mobility(self):
#         """
#         Cập nhật UE mobility và thực hiện quality-based drop.

#         Luồng:
#             1. Di chuyển UE.
#             2. Tính lại distances và gain.
#             3. Kiểm tra best SNR trên toàn bộ RU.
#             4. Drop UE nếu tín hiệu yếu liên tục.
#             5. Bổ sung UE mới để population quay về gần target.

#         Return:
#             removed_ues_with_info:
#                 UE bị drop để caller release resource.

#             id_new_UE:
#                 ID của các UE mới được thêm vào.
#         """
#         removed_ues_with_info = []

#         # =================================================
#         # 1. Mobility + quality-based drop
#         # =================================================
#         for UE_id in list(
#             self.UE_requests.keys()
#         ):
#             UE = self.UE_requests[
#                 UE_id
#             ]

#             mobility_state = UE.setdefault(
#                 "mobility",
#                 self.initialize_mobility_state(),
#             )

#             new_coord = self.adjust_coordinates_UE(
#                 UE[
#                     "coordinate"
#                 ],
#                 mobility_state,
#             )

#             distances, gain = (
#                 self.compute_UE_metrics(
#                     new_coord
#                 )
#             )

#             UE[
#                 "coordinate"
#             ] = new_coord

#             UE[
#                 "distances_RU_UE"
#             ] = distances

#             UE[
#                 "gain"
#             ] = gain

#             should_drop, best_snr_db = (
#                 self.update_drop_state(
#                     UE
#                 )
#             )

#             if should_drop:
#                 print(
#                     f"[DROP] "
#                     f"UE={UE_id} | "
#                     f"BestSNR={best_snr_db:.2f} dB | "
#                     f"Counter="
#                     f"{UE['coverage']['drop_counter']}/"
#                     f"{DROP_TTT_STEPS}"
#                 )
#                 removed_UE = self.remove_UE(
#                     UE_id
#                 )

#                 removed_ues_with_info.append(
#                     (
#                         UE_id,
#                         removed_UE,
#                     )
#                 )

#                 continue

#             UE["status"]["active"] = 1

#         # =================================================
#         # 2. Bổ sung UE mới nếu tải giảm
#         # =================================================
#         id_new_UE = (
#             self.rebalance_UE_population()
#         )
#         print(
#             f"[MOBILITY] "
#             f"UEs={len(self.UE_requests)} | "
#             f"Dropped={len(removed_ues_with_info)} | "
#             f"Added={len(id_new_UE)}"
#         )
#         return (
#             removed_ues_with_info,
#             id_new_UE,
#         )

#     def adjust_coordinates_UE(self, coordinate_UE, mobility_state):
#         """
#             Cập nhật vị trí UE theo correlated random walk.

#             Công thức:
#                 step_distance = speed × delta_t

#                 x_new = x_old + step_distance × cos(direction)
#                 y_new = y_old + step_distance × sin(direction)

#             Trong giai đoạn hiện tại:
#                 - chưa drop UE;
#                 - nếu UE đi quá biên thì phản xạ tạm thời.
#         """
#         # delta_coordinate = 1000
#         x, y = coordinate_UE
#         old_speed = float(mobility_state["speed_mps"])
#         speed_jitter=float(np.random.uniform(- UE_SPEED_JITTER_MPS, UE_SPEED_JITTER_MPS))
#         new_speed = float(np.clip(old_speed + speed_jitter, UE_SPEED_MIN_MPS, UE_SPEED_MAX_MPS))
        
#         # delta_x = np.random.uniform(-delta_coordinate, delta_coordinate)
#         # delta_y = np.random.uniform(-delta_coordinate, delta_coordinate)

#         # new_x = x + delta_x
#         # new_y = y + delta_y

#         # Đưa UE về đúng vùng mô phỏng nếu đi ra ngoài
#         # dist = np.hypot(new_x, new_y)

#         # if dist < self.radius_in:
#         #     if dist < 1e-12:
#         #         angle = np.random.uniform(0, 2 * np.pi)
#         #         new_x = self.radius_in * np.cos(angle)
#         #         new_y = self.radius_in * np.sin(angle)
#         #     else:
#         #         scale = self.radius_in / dist
#         #         new_x *= scale
#         #         new_y *= scale

#         # elif dist > self.radius_out:
#         #     scale = self.radius_out / dist
#         #     new_x *= scale
#         #     new_y *= scale

#         old_direction = float(mobility_state["direction_rad"])
#         direction_jitter = float(np.random.uniform(-UE_DIRECTION_JITTER_RAD, UE_DIRECTION_JITTER_RAD))
#         new_direction = float((old_direction+direction_jitter)%(2.0*np.pi))
        
#         step_distance = new_speed*MOBILITY_TIME_STEP_S
#         new_x = (x + step_distance*np.cos(new_direction))
#         new_y = (y + step_distance*np.sin(new_direction))
        
#         # dist = float(np.hypot(new_x, new_y))
        
#         # if (dist < self.radius_in or dist > self.radius_out):
#         #     # Đảo hướng gần 180 độ
#         #     new_direction = (new_direction + np.pi) % (2.0*np.pi)

#         #     new_x = (x + step_distance*np.cos(new_direction))
#         #     new_y = (y + step_distance*np.sin(new_direction))

#         #     # Guard dự phòng nếu vẫn vượt biên
#         #     dist = float(np.hypot(new_x, new_y))
#         #     if dist > self.radius_out:
#         #         scale = (self.radius_out/max(dist,1e-12))
#         #         new_x *= scale
#         #         new_y *= scale

#         #     elif dist < self.radius_in:
#         #         if dist < 1e-12:
#         #             angle = float(np.random.uniform(0.0,2.0 * np.pi,))
#         #             new_x = (self.radius_in *np.cos(angle))
#         #             new_y = (self.radius_in*np.sin(angle))

#         #         else:
#         #             scale = (self.radius_in /dist)
#         #             new_x *= scale
#         #             new_y *= scale

#         # =================================================
#         # 5. Ghi lại trạng thái mobility
#         # =================================================
#         mobility_state["speed_mps"] = (
#             float(
#                 new_speed
#             )
#         )

#         mobility_state["direction_rad"] = (
#             float(
#                 new_direction
#             )
#         )

#         return (new_x, new_y)

#     def update_UE_request(self, ue_id, update_dict):
#         ue = self.UE_requests[ue_id]

#         for key, value in update_dict.items():
#             # Nếu là dict → update sâu (merge)
#             if isinstance(value, dict) and key in ue and isinstance(ue[key], dict):
#                 ue[key].update(value)
#             else:
#                 # Gán trực tiếp
#                 ue[key] = value

#     def update_drop_state(self, ue_info):
#         """
#         Cập nhật bộ đếm drop của một UE.

#         UE bị drop khi best reference SNR từ tất cả RU
#         thấp hơn ngưỡng trong DROP_TTT_STEPS liên tiếp.

#         Return:
#             should_drop: bool
#             best_snr_db: float
#         """
#         best_snr_db = self.calculate_best_reference_snr_db(ue_info)
#         coverage = ue_info.setdefault(
#             "coverage",
#             {
#                 "drop_counter": 0,
#                 "best_reference_snr_db": None,
#             },
#         )
#         coverage["best_reference_snr_db"] = float(best_snr_db)
#         if (best_snr_db < SNR_DROP_THRESHOLD_DB):
#             coverage["drop_counter"] = int(coverage.get("drop_counter", 0)) + 1
#         else: 
#             coverage["drop_counter"] = 0
        
#         should_drop = (coverage["drop_counter"] >= DROP_TTT_STEPS)
#         return (bool(should_drop), float(best_snr_db))
        
    
#     def check_UE_all_inactive(self):
#         if not self.UE_requests:
#             return True
#         return all(int(ue["status"].get("active", 0)) == 0 for ue in self.UE_requests.values())

#     def rebalance_UE_population(self):
#         """
#         Bổ sung UE mới để giữ population quanh target.

#         Không remove UE ngẫu nhiên.
#         UE chỉ bị remove bởi quality-based drop.
#         """
#         id_new_UE = []

#         current_num_UEs = len(
#             self.UE_requests
#         )

#         # Nếu số UE vẫn đủ cao thì không thêm
#         if (
#             current_num_UEs
#             >=
#             self.target_num_UEs
#         ):
#             return id_new_UE

#         num_to_add = min(
#             MAX_UE_CHURN_PER_STEP,
#             self.target_num_UEs
#             -
#             current_num_UEs,
#         )

#         if num_to_add > 0:
#             id_new_UE.extend(
#                 self.add_UEs_requests(
#                     num_to_add
#                 )
#             )

#         return id_new_UE

class ResourceManager:
    """Quản lý tài nguyên: RB, RU, DU, CU."""

    def __init__(self, num_RBs, num_RUs, num_DUs, num_CUs,bandwidth_per_RB = bandwidth_per_RB, max_RBs_per_UE = max_RBs_per_UE, P_ib_sk_val = P_ib_sk_val, P_i_random_list = P_i_random_list, A_j_random_list = A_j_random_list, A_m_random_list = A_m_random_list, bw_ru_du_random_list = bw_ru_du_random_list, bw_du_cu_random_list = bw_du_cu_random_list):

        # -------------------- Physics --------------------
        self.num_RBs = int(num_RBs)
        self.num_RUs = int(num_RUs)
        self.num_DUs = int(num_DUs)
        self.num_CUs = int(num_CUs)
        self.bandwidth_per_RB = float(bandwidth_per_RB)
        self.max_RBs_per_UE = int(max_RBs_per_UE)
        self.P_ib_sk_val      = list(P_ib_sk_val)
        self.P_i_random_list = P_i_random_list
        self.A_j_random_list = A_j_random_list
        self.A_m_random_list = A_m_random_list
        self.bw_ru_du_random_list = bw_ru_du_random_list
        self.bw_du_cu_random_list = bw_du_cu_random_list
        

        # -------------------- CPU model --------------------
        self.k_DU = float(k_DU)
        self.k_CU = float(k_CU)
        G = self.create_topo()

        self.RU_power_cap, self.DU_cap, self.CU_cap = self.get_node_cap(G)
        self.link_bw_ru_du_bps, self.link_bw_du_cu_bps = self.get_links(G)

        self.coordinates_RU = self.set_coordinates_RU(self.num_RUs)
        self.reset()
        

    def reset(self):
        # -------------------- Remaining resources --------------------
        self.RB_remaining       = int(self.num_RBs)
        self.RU_power_remaining = np.copy(self.RU_power_cap).astype(float)
        self.DU_remaining       = np.copy(self.DU_cap).astype(float)
        self.CU_remaining       = np.copy(self.CU_cap).astype(float)

    def create_topo(self):
        G = nx.Graph()

        # Tạo danh sách các nút RU, DU và CU
        RUs = [f'RU{i+1}' for i in range(self.num_RUs)]
        DUs = [f'DU{i+1}' for i in range(self.num_DUs)]
        CUs = [f'CU{i+1}' for i in range(self.num_CUs)]

        # Thêm các nút RU, DU và CU vào đồ thị
        for ru in RUs:
            G.add_node(ru, type='RU', power = np.random.choice(P_i_random_list))
        for du in DUs:
            G.add_node(du, type='DU', capacity = np.random.choice(A_j_random_list))
        for cu in CUs:
            G.add_node(cu, type='CU', capacity = np.random.choice(A_m_random_list))

        # Kết nối RUs với DUs (Mỗi DU có thể kết nối với tất cả các RU)
        for du in DUs:
            for ru in RUs:
                G.add_edge(ru, du, link_type="RU-DU", bandwidth=np.random.choice(self.bw_ru_du_random_list))

        # Kết nối DUs với CUs (Mỗi DU kết nối với tất cả các CU)
        for du in DUs:
            for cu in CUs:
                G.add_edge(du, cu, link_type="DU-CU", bandwidth=np.random.choice(self.bw_du_cu_random_list))
        return G

    def get_node_cap(self, G):
        ru_weights = []  # Mảng chứa trọng số của các nút RU
        du_weights = []  # Mảng chứa trọng số của các nút DU
        cu_weights = []  # Mảng chứa trọng số của các nút CU

        # Duyệt qua tất cả các nút trong đồ thị
        for node, data in G.nodes(data=True):
            if data['type'] == 'RU':  # Nếu nút là RU
                ru_weights.append(data['power'])
            if data['type'] == 'DU':  # Nếu nút là DU
                du_weights.append(data['capacity'])
            elif data['type'] == 'CU':  # Nếu nút là CU
                cu_weights.append(data['capacity'])

        return ru_weights, du_weights, cu_weights
    
    def get_links(self, G):
        # Lấy danh sách các node theo loại
        RUs = [n for n, d in G.nodes(data=True) if d['type'] == 'RU']
        DUs = [n for n, d in G.nodes(data=True) if d['type'] == 'DU']
        CUs = [n for n, d in G.nodes(data=True) if d['type'] == 'CU']

        # Khởi tạo ma trận băng thông RU–DU và DU–CU (đơn vị bps)
        l_ru_du = np.zeros((len(RUs), len(DUs)))
        l_du_cu = np.zeros((len(DUs), len(CUs)))

        # Duyệt qua các cạnh trong đồ thị
        for u, v, data in G.edges(data=True):
            bw = data.get('bandwidth', 0.0)

            if G.nodes[u]['type'] == 'RU' and G.nodes[v]['type'] == 'DU':
                l_ru_du[RUs.index(u), DUs.index(v)] = bw
            elif G.nodes[u]['type'] == 'DU' and G.nodes[v]['type'] == 'RU':
                l_ru_du[RUs.index(v), DUs.index(u)] = bw
            elif G.nodes[u]['type'] == 'DU' and G.nodes[v]['type'] == 'CU':
                l_du_cu[DUs.index(u), CUs.index(v)] = bw
            elif G.nodes[u]['type'] == 'CU' and G.nodes[v]['type'] == 'DU':
                l_du_cu[DUs.index(v), CUs.index(u)] = bw

        return l_ru_du, l_du_cu

    def set_coordinates_RU(self, num_RUs, radius_out = 1000):
        circle_RU_out = radius_out * 0.65
        angles = np.linspace(0, 2 * np.pi, num_RUs - 1, endpoint=False) 
        x = np.concatenate(([0], circle_RU_out * np.cos(angles)))  
        y = np.concatenate(([0], circle_RU_out * np.sin(angles)))  
        coordinates_RU = list(zip(x, y)) 
        return coordinates_RU
    
    def build_RAN_snapshot(self):
        RAN_snapshot = {
            "RB_remaining":       int(self.RB_remaining),
            "RU_power_remaining": np.copy(self.RU_power_remaining),
            "DU_remaining":       np.copy(self.DU_remaining),
            "CU_remaining":       np.copy(self.CU_remaining),
            "link_bw_ru_du_bps":  np.copy(self.link_bw_ru_du_bps),
            "link_bw_du_cu_bps":  np.copy(self.link_bw_du_cu_bps),
        }
        return RAN_snapshot

    def update_resources(self, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc, cpu_DU_req, cpu_CU_req):
        self.RB_remaining -= int(num_RB_alloc)
        self.RU_power_remaining[int(RU_choice)] -= float(power_alloc)
        self.DU_remaining[int(DU_choice)] -= float(cpu_DU_req)
        self.CU_remaining[int(CU_choice)] -= float(cpu_CU_req)
        

    def release_resources(self, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc, cpu_DU_req, cpu_CU_req):
        self.RB_remaining += int(num_RB_alloc)
        self.RU_power_remaining[int(RU_choice)] += float(power_alloc)
        self.DU_remaining[int(DU_choice)] += float(cpu_DU_req)
        self.CU_remaining[int(CU_choice)] += float(cpu_CU_req)
        
# =======================================================
# ====================== ENV ============================
# =======================================================
class HandOverEnv:
    def __init__(self, num_UEs, num_RBs, total_nodes, num_RUs, num_DUs, num_CUs, radio_log_path=None):

        self.w_acc = 1/4        # Trọng số accept
        self.w_thr = 1/4        # Trọng số throughtput
        self.w_lat = 1/4        # Trọng số latency
        self.w_handover = 1/4   # Trọng số handover

        # -------------------- Sizes & Presets --------------------
        self.num_UEs        = int(num_UEs)
        self.num_RBs        = int(num_RBs)
        self.total_nodes    = int(total_nodes)
        self.num_RUs        = int(num_RUs)
        self.num_DUs        = int(num_DUs)
        self.num_CUs        = int(num_CUs)

        # -------------------- CPU model (cycles/bit) --------------------
        self.k_DU = float(k_DU)
        self.k_CU = float(k_CU)

        # -------------------- Managers --------------------    
        self.resource_manager = ResourceManager(self.num_RBs, self.num_RUs, self.num_DUs, self.num_CUs)
        self.UE_manager = UEManager(self.resource_manager.coordinates_RU, target_num_UEs=self.num_UEs)
        self.resource_manager.reset()
        _ = self.UE_manager.add_UEs_requests(self.num_UEs)
        # =====================================================
        # Radio metric logger
        # =====================================================
        self.radio_debug_step = 0
        self.radio_log_file = None
        self.radio_log_writer = None

        if radio_log_path is not None:
            radio_log_path = Path(radio_log_path).resolve()
            radio_log_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.radio_log_path = str(radio_log_path)

            self.radio_log_file = open(
                self.radio_log_path,
                mode="w",
                newline="",
                encoding="utf-8",
                buffering=1,
            )

            self.radio_log_writer = csv.DictWriter(
                self.radio_log_file,
                fieldnames = [
                    "step",
                    "ue_id",
                    "slice_type",
                    "ru_choice",
                    "num_rb_alloc",
                    "slice_max_rb",
                    "power_alloc_w",
                    "power_per_rb_w",
                    "snr_linear",
                    "snr_db",
                    "throughput_mbps",
                    "r_min_mbps",
                    "success",
                    "reason",

                    # Thông tin tài nguyên RU hiện tại
                    "ru_power_remaining_w",

                    # Giá trị tối đa tại RU đang được chọn
                    "max_rb_at_selected_ru",
                    "max_power_at_selected_ru_w",
                    "max_snr_at_selected_ru_db",
                    "max_throughput_at_selected_ru_mbps",

                    # RU tốt nhất trong trạng thái hiện tại
                    "best_ru",
                    "best_ru_snr_db",
                    "best_ru_max_throughput_mbps",
                ],  
            )

            self.radio_log_writer.writeheader()
            self.radio_log_file.flush()

            print(
                f"[RADIO LOG] File created at: "
                f"{self.radio_log_path}"
            )
        else:
            print(
                "[RADIO LOG] Disabled: "
                "radio_log_path is None"
            )
    def build_radio_metrics(
        self,
        UE_id,
        UE,
        RU_choice,
        num_RB_alloc,
        power_alloc,
    ):
        """
        Thu thập radio metrics của action hiện tại.

        Lưu ý:
            Môi trường hiện chưa xét interference giữa các RU.
            Vì vậy SINR được xấp xỉ bằng SNR.
        """
        eps = 1e-12

        gain_list = np.asarray(
            UE.get("gain", []),
            dtype=float,
        )

        if RU_choice < 0 or RU_choice >= len(gain_list):
            return None

        num_RB_alloc = max(
            1,
            int(num_RB_alloc),
        )

        power_alloc = float(power_alloc)

        bandwidth_per_RB = float(
            self.resource_manager.bandwidth_per_RB
        )

        gain = float(gain_list[RU_choice])

        # =====================================================
        # Metrics của action thực tế
        # =====================================================
        power_per_RB = power_alloc / num_RB_alloc

        snr_linear = (
            power_per_RB
            * gain
        )

        snr_db = 10.0 * np.log10(
            max(snr_linear, eps)
        )

        throughput_bps = (
            num_RB_alloc
            * bandwidth_per_RB
            * np.log2(
                1.0 + snr_linear
            )
        )

        # =====================================================
        # RB limit theo slice
        # =====================================================
        slice_max_RB = int(
            UE.get(
                "max_RBs",
                self.resource_manager.max_RBs_per_UE,
            )
        )

        rb_remaining = int(
            self.resource_manager.RB_remaining
        )

        ru_power_remaining = float(
            self.resource_manager.RU_power_remaining[
                RU_choice
            ]
        )

        max_action_power = float(
            np.max(
                self.resource_manager.P_ib_sk_val
            )
        )

        max_RB_effective = max(
            1,
            min(
                slice_max_RB,
                rb_remaining,
            ),
        )

        max_power_effective = max(
            0.0,
            min(
                max_action_power,
                ru_power_remaining,
            ),
        )

        max_power_per_RB = (
            max_power_effective
            / max_RB_effective
        )

        max_snr_linear = (
            max_power_per_RB
            * gain
        )

        max_snr_db = 10.0 * np.log10(
            max(max_snr_linear, eps)
        )

        max_throughput_bps = (
            max_RB_effective
            * bandwidth_per_RB
            * np.log2(
                1.0 + max_snr_linear
            )
        )

        # =====================================================
        # RU tốt nhất theo throughput cực đại
        # =====================================================
        best_RU = None
        best_RU_snr_db = None
        best_RU_max_throughput_bps = -1.0

        for ru_idx, ru_gain in enumerate(gain_list):
            ru_power_available = float(
                self.resource_manager.RU_power_remaining[
                    ru_idx
                ]
            )

            ru_max_power = max(
                0.0,
                min(
                    max_action_power,
                    ru_power_available,
                ),
            )

            ru_power_per_RB = (
                ru_max_power
                / max_RB_effective
            )

            ru_snr_linear = (
                ru_power_per_RB
                * float(ru_gain)
            )

            ru_throughput_bps = (
                max_RB_effective
                * bandwidth_per_RB
                * np.log2(
                    1.0 + ru_snr_linear
                )
            )

            if ru_throughput_bps > best_RU_max_throughput_bps:
                best_RU = int(ru_idx)

                best_RU_snr_db = 10.0 * np.log10(
                    max(ru_snr_linear, eps)
                )

                best_RU_max_throughput_bps = (
                    ru_throughput_bps
                )

        return {
            "step": int(self.radio_debug_step),
            "ue_id": int(UE_id),
            "slice_type": UE.get(
                "type",
                UE.get("slice_type", "unknown"),
            ),
            "ru_choice": int(RU_choice),
            "num_rb_alloc": int(num_RB_alloc),
            "slice_max_rb": int(slice_max_RB),
            "power_alloc_w": float(power_alloc),
            "power_per_rb_w": float(power_per_RB),
            "snr_linear": float(snr_linear),
            "snr_db": float(snr_db),
            "throughput_mbps": float(
                throughput_bps / 1e6
            ),
            "r_min_mbps": float(
                UE["R_min"] / 1e6
            ),
            "ru_power_remaining_w": float(
                ru_power_remaining
            ),
            "max_rb_at_selected_ru": int(
                max_RB_effective
            ),
            "max_power_at_selected_ru_w": float(
                max_power_effective
            ),
            "max_snr_at_selected_ru_db": float(
                max_snr_db
            ),
            "max_throughput_at_selected_ru_mbps": float(
                max_throughput_bps / 1e6
            ),
            "best_ru": int(best_RU),
            "best_ru_snr_db": float(
                best_RU_snr_db
            ),
            "best_ru_max_throughput_mbps": float(
                best_RU_max_throughput_bps / 1e6
            ),
        }
    def write_radio_metrics(
        self,
        metrics,
        success,
        reason,
    ):
        """
        Ghi một dòng radio metrics xuống file CSV.
        Flush ngay sau mỗi dòng để tránh mất dữ liệu nếu chương trình dừng đột ngột.
        """
        if not DEBUG_RADIO_METRICS:
            return

        if metrics is None:
            return

        if (
            DEBUG_RADIO_UE_ID is not None
            and int(metrics["ue_id"])
            != int(DEBUG_RADIO_UE_ID)
        ):
            return

        metrics = dict(metrics)

        metrics["success"] = bool(success)
        metrics["reason"] = str(reason)

        self.radio_log_writer.writerow(metrics)

        # Đảm bảo dữ liệu được đẩy ngay xuống file
        self.radio_log_file.flush()

        print(
            f"[RADIO] "
            f"UE={metrics['ue_id']} | "
            f"Slice={metrics['slice_type']} | "
            f"RU={metrics['ru_choice']} | "
            f"RB={metrics['num_rb_alloc']}/"
            f"{metrics['slice_max_rb']} | "
            f"SNR={metrics['snr_db']:.2f} dB | "
            f"R={metrics['throughput_mbps']:.2f} Mbps | "
            f"R_min={metrics['r_min_mbps']:.2f} Mbps | "
            f"R_max={metrics['max_throughput_at_selected_ru_mbps']:.2f} Mbps | "
            f"Success={metrics['success']} | "
            f"Reason={metrics['reason']}"
        )
    def write_radio_metrics_csv(
        self,
        UE_id,
        UE,
        RU_choice,
        num_RB_alloc,
        power_alloc,
        success,
        reason,
    ):
        """
        Ghi thông tin radio của từng action xuống file CSV.

        Lưu ý:
            Môi trường hiện chưa xét interference giữa các RU.
            Do đó SINR đang được xấp xỉ bằng SNR.
        """
        eps = 1e-12

        # =====================================================
        # 1. Đường dẫn log tuyệt đối
        # =====================================================
        log_path = Path("./logs/radio_metrics.csv").resolve()

        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # 2. Kiểm tra gain và RU
        # =====================================================
        gain_list = np.asarray(
            UE.get("gain", []),
            dtype=float,
        )

        if RU_choice < 0 or RU_choice >= len(gain_list):
            return

        num_RB_alloc = max(
            1,
            int(num_RB_alloc),
        )

        power_alloc = float(power_alloc)

        # =====================================================
        # 3. Tính SNR và throughput thực tế
        # =====================================================
        bandwidth_per_RB = float(
            self.resource_manager.bandwidth_per_RB
        )

        gain = float(
            gain_list[RU_choice]
        )

        power_per_RB = (
            power_alloc
            / num_RB_alloc
        )

        snr_linear = (
            power_per_RB
            * gain
        )

        snr_db = 10.0 * np.log10(
            max(
                snr_linear,
                eps,
            )
        )

        throughput_bps = (
            num_RB_alloc
            * bandwidth_per_RB
            * np.log2(
                1.0
                +
                snr_linear
            )
        )

        # =====================================================
        # 4. Tính throughput tối đa tại RU hiện tại
        #    Giữ nguyên max_RBs_per_UE = 10
        # =====================================================
        max_rb = int(
            self.resource_manager.max_RBs_per_UE
        )

        ru_power_remaining = float(
            self.resource_manager.RU_power_remaining[
                RU_choice
            ]
        )

        max_power_level = float(
            np.max(
                self.resource_manager.P_ib_sk_val
            )
        )

        max_power_effective = min(
            max_power_level,
            ru_power_remaining,
        )

        max_power_per_RB = (
            max_power_effective
            /
            max_rb
        )

        max_snr_linear = (
            max_power_per_RB
            *
            gain
        )

        max_snr_db = 10.0 * np.log10(
            max(
                max_snr_linear,
                eps,
            )
        )

        max_throughput_bps = (
            max_rb
            *
            bandwidth_per_RB
            *
            np.log2(
                1.0
                +
                max_snr_linear
            )
        )

        # =====================================================
        # 5. Tạo header và dữ liệu
        # =====================================================
        fieldnames = [
            "ue_id",
            "slice_type",
            "ru_choice",
            "num_rb_alloc",
            "max_rb",
            "power_alloc_w",
            "power_per_rb_w",
            "snr_linear",
            "snr_db",
            "throughput_mbps",
            "r_min_mbps",
            "ru_power_remaining_w",
            "max_power_effective_w",
            "max_snr_db",
            "max_throughput_mbps",
            "success",
            "reason",
        ]

        row = {
            "ue_id": int(UE_id),
            "slice_type": UE.get(
                "type",
                UE.get(
                    "slice_type",
                    "unknown",
                ),
            ),
            "ru_choice": int(RU_choice),
            "num_rb_alloc": int(
                num_RB_alloc
            ),
            "max_rb": int(
                max_rb
            ),
            "power_alloc_w": float(
                power_alloc
            ),
            "power_per_rb_w": float(
                power_per_RB
            ),
            "snr_linear": float(
                snr_linear
            ),
            "snr_db": float(
                snr_db
            ),
            "throughput_mbps": float(
                throughput_bps
                /
                1e6
            ),
            "r_min_mbps": float(
                UE["R_min"]
                /
                1e6
            ),
            "ru_power_remaining_w": float(
                ru_power_remaining
            ),
            "max_power_effective_w": float(
                max_power_effective
            ),
            "max_snr_db": float(
                max_snr_db
            ),
            "max_throughput_mbps": float(
                max_throughput_bps
                /
                1e6
            ),
            "success": bool(
                success
            ),
            "reason": str(
                reason
            ),
        }

        # =====================================================
        # 6. Ghi file ngay lập tức
        # =====================================================
        file_exists = log_path.exists()

        with open(
            log_path,
            mode="a",
            newline="",
            encoding="utf-8",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)

            # Đẩy dữ liệu xuống ổ đĩa ngay
            f.flush()

        # print(
        #     f"[RADIO LOG] Written to: {log_path}"
        # )
    # ======================================================================
    # Basics
    # ======================================================================
    def get_state(self):
        snap_short_RAN = self.resource_manager.build_RAN_snapshot()
        snap_short_UE_request = self.UE_manager.UE_requests
        state = {
            "RAN": snap_short_RAN,
            "UE_requests": snap_short_UE_request,
        }
        return state
    
    def calculate_latency(self, distances_RU_UE, throughput_bps, packet_size_bits, cycles_per_packet, lambda_default_pps, cpu_DU_req, cpu_CU_req):
        """
        Tính độ trễ đầu-cuối của một UE.
        Thành phần độ trễ:
            1. Lan truyền trên liên kết RU-UE
            2. Truyền dữ liệu
            3. Chờ trong hàng đợi tại DU và CU
            4. Xử lý tại DU và CU

        Mô hình hàng đợi:
            M/M/1 tại DU và CU

        Đơn vị đầu ra:
            giây (s)
        """
        eps = 1e-12
        c_speed = 3e8  # Tốc độ ánh sáng [m/s]

        # Ép kiểu để tránh lỗi do truyền vào numpy scalar hoặc int
        throughput_bps = float(throughput_bps)
        packet_size_bits = float(packet_size_bits)
        cycles_per_packet = float(cycles_per_packet)
        lambda_pps = float(lambda_default_pps)
        cpu_DU_req = float(cpu_DU_req)
        cpu_CU_req = float(cpu_CU_req)

        # -----------------------------------------------------
        # 1. Propagation latency: RU -> UE
        # -----------------------------------------------------
        L_prop = float(distances_RU_UE) / c_speed

        # Không thể truyền dữ liệu nếu throughput không hợp lệ
        if throughput_bps <= eps:
            return float("inf"), {
                "prop": L_prop,
                "trans": float("inf"),
                "queue_du": float("inf"),
                "queue_cu": float("inf"),
                "queu": float("inf"),
                "proc_du": float("inf"),
                "proc_cu": float("inf"),
                "total": float("inf"),
                "reason": "invalid throughput",
            }

        # -----------------------------------------------------
        # 2. Transmission latency
        # -----------------------------------------------------
        L_trans = packet_size_bits / throughput_bps

        # Không thể xác định tốc độ phục vụ nếu số cycle/gói không hợp lệ
        if cycles_per_packet <= eps:
            return float("inf"), {
                "prop": L_prop,
                "trans": L_trans,
                "queue_du": float("inf"),
                "queue_cu": float("inf"),
                "queu": float("inf"),
                "proc_du": float("inf"),
                "proc_cu": float("inf"),
                "total": float("inf"),
                "reason": "invalid cycles_per_packet",
            }

        # -----------------------------------------------------
        # 3. Service rate tại DU và CU [packet/s]
        # -----------------------------------------------------
        mu_du = cpu_DU_req / cycles_per_packet
        mu_cu = cpu_CU_req / cycles_per_packet

        # Điều kiện ổn định của hàng đợi M/M/1:
        # lambda < mu
        if mu_du <= lambda_pps + eps or mu_cu <= lambda_pps + eps:
            return float("inf"), {
                "prop": L_prop,
                "trans": L_trans,
                "queue_du": float("inf"),
                "queue_cu": float("inf"),
                "queu": float("inf"),
                "proc_du": float("inf"),
                "proc_cu": float("inf"),
                "total": float("inf"),
                "reason": "queue overload",
            }

        # -----------------------------------------------------
        # 4. Queuing latency theo mô hình M/M/1
        # -----------------------------------------------------
        rho_du = lambda_pps / mu_du
        rho_cu = lambda_pps / mu_cu

        L_queue_du = rho_du / (mu_du - lambda_pps)
        L_queue_cu = rho_cu / (mu_cu - lambda_pps)

        # -----------------------------------------------------
        # 5. Processing latency
        # -----------------------------------------------------
        L_proc_du = 1.0 / mu_du
        L_proc_cu = 1.0 / mu_cu

        # -----------------------------------------------------
        # 6. Total E2E latency
        # -----------------------------------------------------
        L_queue_total = L_queue_du + L_queue_cu

        total_latency = (
            L_prop
            + L_trans
            + L_queue_total
            + L_proc_du
            + L_proc_cu
        )

        latency_path = {
            "prop": L_prop,
            "trans": L_trans,
            "queu": L_queue_total,
            "queue_du": L_queue_du,
            "queue_cu": L_queue_cu,
            "proc_du": L_proc_du,
            "proc_cu": L_proc_cu,
            "mu_du": mu_du,
            "mu_cu": mu_cu,
            "rho_du": rho_du,
            "rho_cu": rho_cu,
            "total": total_latency,
        }

        return total_latency, latency_path
    
    def calculate_throughput(self, gain, power_level_alloc, num_RB_alloc):
        power_per_RB = power_level_alloc / num_RB_alloc
        SNR_per_RB   = power_per_RB * gain
        throughput_bps = num_RB_alloc * self.resource_manager.bandwidth_per_RB * np.log2(1.0 + SNR_per_RB)
        return throughput_bps

    def compute_reward(self, UE, throughput_UE, latency_UE, is_handover, is_pingpong):
        R_min = max(float(UE.get("R_min", 1e-9)), 1e-9)
        D_max = max(float(UE.get("delay", 1e-9)), 1e-9)

        normalize_throughput = throughput_UE / R_min
        normalize_latency  = D_max / latency_UE

        w_acc_s = float(UE.get("weight_accept", 1.0))
        w_thr_s = float(UE.get("weight_throughput", 1.0))
        w_lat_s = float(UE.get("weight_latency", 1.0))
        w_HO_s = float(UE.get("weight_handover", 1.0))


        acc_term = self.w_acc * w_acc_s * 1.0
        thr_term = self.w_thr * w_thr_s * normalize_throughput
        lat_term = self.w_lat * w_lat_s * normalize_latency
        handover_term = self.w_handover * w_HO_s * is_handover * (1 + np.exp(is_pingpong))

        # if handover_term != 0:
        #     print(f"handover_term = {handover_term}")

        reward = acc_term + thr_term + lat_term - handover_term

        return float(reward), float(acc_term), float(thr_term), float(lat_term), float(-handover_term)

    
    def compute_resource_allocation(self, UE, RU_choice, num_RB_alloc, power_alloc):
        """
        Trả về:
            throughput (bps)
            latency (s)
            cpu_DU_req (cycles/s)
            cpu_CU_req (cycles/s)
        """
        gain_UE = UE['gain'][RU_choice]
        distances_RU_UE = UE['distances_RU_UE'][RU_choice]
        packet_size_bits = UE['packet_size_bits']
        cycles_per_packet = UE['cycles_per_packet']
        lambda_default_pps = UE['lambda_default_pps']

        throughput_UE = self.calculate_throughput(gain_UE, power_alloc, num_RB_alloc)

        # Calculate CPU requirement at DU and CU
        cpu_DU_req =  self.k_DU * throughput_UE * (1 + UE['eta_slice'])
        cpu_CU_req =  self.k_CU * throughput_UE * (1 + UE['eta_slice'])

        latency_UE, latency_path_UE = self.calculate_latency(distances_RU_UE, throughput_UE, packet_size_bits, cycles_per_packet, lambda_default_pps, cpu_DU_req, cpu_CU_req)

        return throughput_UE, latency_UE, cpu_DU_req, cpu_CU_req
    
    def check_feasible(self, UE, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc):

        eps = 1e-9  
        if num_RB_alloc >  self.resource_manager.RB_remaining:
           
            return False, "insufficient RB resource"
        
        if power_alloc > self.resource_manager.RU_power_remaining[RU_choice]:
            
            return False, "insufficient RU resource"
        
        throughput_UE, lantecy_UE, cpu_DU_req_UE, cpu_CU_req_UE = self.compute_resource_allocation(UE, RU_choice, num_RB_alloc, power_alloc)

        if throughput_UE + eps < UE["R_min"]:
            
            return False, f"insufficient_throughput ({throughput_UE:.2f} < {UE['R_min']:.2f})"
        
        if lantecy_UE > UE["delay"]:
            
            return False, f"latency_violation ({lantecy_UE:.6f}s > {UE['delay']:.6f}s)"
        
        # ---- DU/CU CPU budgets (cycles/s) ----
        if self.resource_manager.DU_remaining[int(DU_choice)] + 1e-9 < float(cpu_DU_req_UE):
            
            return False, "insufficient DU resource"
        
        if self.resource_manager.CU_remaining[int(CU_choice)] + 1e-9 < float(cpu_CU_req_UE):
           
            return False, "insufficient CU resource"
        
        return True, (throughput_UE, lantecy_UE, cpu_DU_req_UE, cpu_CU_req_UE)


    def check_action_valid(self, UE_id, handover_flag, RU, DU, CU, RB, power):
        UE = self.UE_manager.UE_requests[UE_id]
        if UE_id not in self.UE_manager.UE_requests:
            return False, 
        
        if handover_flag not in [0, 1]:
            return False

        if not (0 <= RU < self.num_RUs):
            return False

        if not (0 <= DU < self.num_DUs):
            return False

        if not (0 <= CU < self.num_CUs):
            return False

        # if RB < 0 or RB > self.resource_manager.max_RBs_per_UE:
        #     return False
        slice_max_RBs = int(
            UE.get(
                "max_RBs",
                self.resource_manager.max_RBs_per_UE,
            )
        )

        if RB < 1 or RB > slice_max_RBs:
            return False
        
        if RB > self.resource_manager.RB_remaining:
            return False

        if power not in self.resource_manager.P_ib_sk_val:
            return False
        
        return True

    # ======================================================================
    # RL step
    # ======================================================================
    def step(self, action):
        """
        action = (UE_id, handover_flag, ru_sel, du_sel, cu_sel, num_RB_alloc, power_alloc)
        returns: (state, reward, done, info)
        """
        #print(action)
        UE_idx        = int(action[0])
        handover_flag = int(action[1])
        RU_choice     = int(action[2])
        DU_choice     = int(action[3])
        CU_choice     = int(action[4])
        num_RB_alloc   = int(action[5])
        power_alloc    = float(action[6])
        
        UE = self.UE_manager.UE_requests[UE_idx]
        prev_alloc = copy.deepcopy(UE["allocation"])
        prev_prev_alloc = copy.deepcopy(UE["prev_allocation"])

        prev_RU = prev_alloc["RU"]
        prev_DU = prev_alloc["DU"]
        prev_CU = prev_alloc["CU"]
        prev_RB = prev_alloc["num_RB_alloc"]
        prev_power = prev_alloc["power_alloc"]
        prev_cpu_DU = prev_alloc["cpu_DU_req"]
        prev_cpu_CU = prev_alloc["cpu_CU_req"]

        old_alloc_exists = prev_RU is not None

        # Nếu UE này đã được mapping trước đó 
        if old_alloc_exists:
            # Vì UE này đã được mapping trướ đó nên ta cần giải phóng tài nguyên mà UE đã dùng trước đó để tính toán lại tài nguyên phân bổ
            # Nếu không giải phóng trước thì tài nguyên sử dụng trước đó và tài nguyên cấp cho thời điểm hiện tại sẽ bị chồng lấn lên nhau
            self.resource_manager.release_resources(prev_RU, prev_DU, prev_CU, prev_RB, prev_power, prev_cpu_DU, prev_cpu_CU)

            valid = self.check_action_valid(UE_idx, handover_flag, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)
            if not valid:
                is_pingpong = 0
                is_handover = 0
                reward = 0
                self.UE_manager.update_UE_request(UE_idx, {
                    "status": {"active": 0, "served": False, "reason": "Invalid action, choice RU, DU, CU other"},
                    "allocation" : self.UE_manager.empty_alloc(),
                    "prev_allocation": prev_alloc,
                    "pingpong": is_pingpong,
                    "handover_count": is_handover,
                })

                done = self.check_done()
                return self.get_state(), reward, done, {"success": False, "reason": "Invalid action, choice RU, DU, CU other"}
            
        
            # =========================================================
            # CASE 1: handover_flag = 0 và tồn tại allocation trước đó -> (UE đã được mapping trước đó và tiếp tục giữ nguyên mapping)
            # =========================================================
            if handover_flag == 0:
                # Kiểm tra xem có thực sự là giữ nguyên mapping hay không
                if RU_choice == prev_RU and DU_choice == prev_DU and CU_choice == prev_CU:
                    is_handover = 0
                    is_pingpong = 0
                    # Check xem việc quyết định mapping và phân bổ tài nguyên có đảm bảo các điều kiện ràng buộc
                    # radio_metrics = self.build_radio_metrics(
                    #     UE_id=UE_idx,
                    #     UE=UE,
                    #     RU_choice=RU_choice,
                    #     num_RB_alloc=num_RB_alloc,
                    #     power_alloc=power_alloc,
                    # )
                    # Check xem việc quyết định mapping và phân bổ tài nguyên có đảm bảo các điều kiện ràng buộc
                    feasible, msg = self.check_feasible(UE, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)
                    self.write_radio_metrics_csv(
                        UE_id=UE_idx,
                        UE=UE,
                        RU_choice=RU_choice,
                        num_RB_alloc=num_RB_alloc,
                        power_alloc=power_alloc,
                        success=feasible,
                        reason=msg,
                    )
                    # self.write_radio_metrics(
                    #     metrics=radio_metrics,
                    #     success=feasible,
                    #     reason=msg,
                    # )
                    if not feasible:
                        reward = 0
                        self.UE_manager.update_UE_request(UE_idx, {
                            "status": {"active": 0, "served": False, "reason": "keep_mapping_not_valid"},
                            "allocation" : self.UE_manager.empty_alloc(),
                            "prev_allocation": prev_alloc,
                            "pingpong": is_pingpong,
                            "handover_count": is_handover,
                        })

                        done = self.check_done()
    
                        return self.get_state(), reward, done, {"success": False, "reason": msg}
                        # Thoả mãn
                    else:
                        throughput_UE, latency_UE, cpu_DU_req, cpu_CU_req = msg[0], msg[1], msg[2], msg[3]
                        # Mạng RAN cần cấp tài nguyên cho UE đó
                        self.resource_manager.update_resources(RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc, cpu_DU_req, cpu_CU_req)

                        # Cập nhật trạng thái cho UE
                        self.UE_manager.update_UE_request(UE_idx, {
                            "status": {"active": 0, "served": True, "reason": "accepted_success"},
                            "allocation": {
                                "RU": RU_choice,
                                "DU": DU_choice,
                                "CU": CU_choice,
                                "num_RB_alloc": num_RB_alloc,
                                "power_alloc": power_alloc,
                                "throughput_bps": throughput_UE,
                                "delay_s": latency_UE,
                                "cpu_DU_req": cpu_DU_req,
                                "cpu_CU_req": cpu_CU_req,
                            },
                            "prev_allocation": prev_alloc,
                            "pingpong": is_pingpong,
                            "handover_count": is_handover,
                        })

                        # Tính toán reward từ việc phân bổ
                        reward, acc_term, thr_term, lat_term, handover_term = self.compute_reward(UE, throughput_UE, latency_UE, is_handover, is_pingpong)

                        done = self.check_done()
                        
                        return self.get_state(), reward, done, {
                            "success": True,
                            "reason": "UE accepted_success: keep mapping",
                            "throughput_UE": throughput_UE,
                            "latency_UE": latency_UE,
                            "handover": is_handover,
                            "pingpong": is_pingpong,
                            "acc_term": acc_term,
                            "thr_term": thr_term,
                            "lat_term": lat_term,
                            "handover_term": handover_term,
                        }
                
                # Nếu keep mapping mà không phải keep mapping -> Action không hợp lệ
                else: 
                    is_pingpong = 0
                    is_handover = 0
                    reward = 0
                    self.UE_manager.update_UE_request(UE_idx, {
                        "status": {"active": 0, "served": False, "reason": "Invalid action, choice RU, DU, CU other"},
                        "allocation" : self.UE_manager.empty_alloc(),
                        "prev_allocation": prev_alloc,
                        "pingpong": is_pingpong,
                        "handover_count": is_handover,
                    })

                    done = self.check_done()
                    
                    return self.get_state(), reward, done, {"success": False, "reason": "Invalid action, choice RU, DU, CU other"}
            
            # =========================================================
            # CASE 2: handover_flag = 1 và UE có tồn tại trạng thái trước đó
            # =========================================================
            else:
                # Check: nếu là handover thật
                if RU_choice != prev_RU:
                    # Check xem việc quyết định mapping và phân bổ tài nguyên có đảm bảo các điều kiện ràng buộc
                    # radio_metrics = self.build_radio_metrics(
                    #     UE_id=UE_idx,
                    #     UE=UE,
                    #     RU_choice=RU_choice,
                    #     num_RB_alloc=num_RB_alloc,
                    #     power_alloc=power_alloc,
                    # )
                    # Check xem việc quyết định mapping và phân bổ tài nguyên có đảm bảo các điều kiện ràng buộc
                    feasible, msg = self.check_feasible(UE, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)
                    self.write_radio_metrics_csv(
                        UE_id=UE_idx,
                        UE=UE,
                        RU_choice=RU_choice,
                        num_RB_alloc=num_RB_alloc,
                        power_alloc=power_alloc,
                        success=feasible,
                        reason=msg,
                    )
                    # self.write_radio_metrics(
                    #     metrics=radio_metrics,
                    #     success=feasible,
                    #     reason=msg,
                    # )

                    # Nếu handover sang RU mới không đảm bảo ràng buộc
                    if not feasible:
                        reward = 0
                        is_pingpong = 0
                        is_handover = 0
                        self.UE_manager.update_UE_request(UE_idx, {
                            "status": {"active": 0, "served": False, "reason": "Handover but not ensure resource"},
                            "allocation" : self.UE_manager.empty_alloc(),
                            "prev_allocation": prev_alloc,
                            "pingpong": is_pingpong,
                            "handover_count": is_handover,
                        })

                        done = self.check_done()
     
                        return self.get_state(), reward, done, {"success": False, "reason": "Handover butnot ensure resource"}
                    
                    # Nếu đảm bảo ràng buộc
                    else: 
                        is_handover = 1
                        throughput_UE, latency_UE, cpu_DU_req, cpu_CU_req = msg[0], msg[1], msg[2], msg[3]
                        # Mạng RAN cần cấp tài nguyên cho UE đó
                        self.resource_manager.update_resources(RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc, cpu_DU_req, cpu_CU_req)

                        # Check xem handover có tạo ra pingpong hay không
                        prev_prev_RU_choice = prev_prev_alloc['RU']

                        # Nếu pingpong
                        if RU_choice == prev_prev_RU_choice:
                            is_pingpong = UE['pingpong'] + 1
                            reward, acc_term, thr_term, lat_term, handover_term = self.compute_reward(UE, throughput_UE, latency_UE, is_handover, is_pingpong)

                            # Cập nhật trạng thái cho UE
                            self.UE_manager.update_UE_request(UE_idx, {
                                "status": {"active": 0, "served": True, "reason": "accepted_success"},
                                "allocation": {
                                    "RU": RU_choice,
                                    "DU": DU_choice,
                                    "CU": CU_choice,
                                    "num_RB_alloc": num_RB_alloc,
                                    "power_alloc": power_alloc,
                                    "throughput_bps": throughput_UE,
                                    "delay_s": latency_UE,
                                    "cpu_DU_req": cpu_DU_req,
                                    "cpu_CU_req": cpu_CU_req,
                                },
                                "prev_allocation": prev_alloc,
                                "pingpong": is_pingpong,
                                "handover_count": UE['handover_count'] + is_handover,
                            })

                            done = self.check_done()
                    
                            return self.get_state(), reward, done, {
                                "success": True,
                                "reason": "UE accept success: Handover, pingpong",
                                "throughput_UE": throughput_UE,
                                "latency_UE": latency_UE,
                                "handover": is_handover,
                                "pingpong": is_pingpong,
                                "acc_term": acc_term,
                                "thr_term": thr_term,
                                "lat_term": lat_term,
                                "handover_term": handover_term,
                            }
                        
                        # Nếu không pingpong
                        else:
                            is_pingpong = 0
                            reward, acc_term, thr_term, lat_term, handover_term = self.compute_reward(UE, throughput_UE, latency_UE, is_handover, is_pingpong)

                            # Cập nhật trạng thái cho UE
                            self.UE_manager.update_UE_request(UE_idx, {
                                "status": {"active": 0, "served": True, "reason": "accepted_success: Handover, not pingpong"},
                                "allocation": {
                                    "RU": RU_choice,
                                    "DU": DU_choice,
                                    "CU": CU_choice,
                                    "num_RB_alloc": num_RB_alloc,
                                    "power_alloc": power_alloc,
                                    "throughput_bps": throughput_UE,
                                    "delay_s": latency_UE,
                                    "cpu_DU_req": cpu_DU_req,
                                    "cpu_CU_req": cpu_CU_req,
                                },
                                "prev_allocation": prev_alloc,
                                "pingpong": is_pingpong,
                                "handover_count": UE['handover_count'] + is_handover,
                            })

                            done = self.check_done()
                            
                            return self.get_state(), reward, done, {
                                "success": True,
                                "reason": "UE accept success: Handover, not pingpong",
                                "throughput_UE": throughput_UE,
                                "latency_UE": latency_UE,
                                "handover": is_handover,
                                "pingpong": is_pingpong,
                                "acc_term": acc_term,
                                "thr_term": thr_term,
                                "lat_term": lat_term,
                                "handover_term": handover_term,
                            }

                # Handover mà không phải handover
                else: 
                    is_pingpong = 0
                    is_handover = 0
                    reward = 0
                    self.UE_manager.update_UE_request(UE_idx, {
                        "status": {"active": 0, "served": False, "reason": "Invalid action: Is handover but not handover"},
                        "allocation" : self.UE_manager.empty_alloc(),
                        "prev_allocation": prev_alloc,
                        "pingpong": is_pingpong,
                        "handover_count": is_handover,
                    })

                    done = self.check_done()
                    
                    return self.get_state(), reward, done, {"success": False, "reason": "Invalid action: Is handover but not handover"}
                
        # Nếu UE này chưa được mapping trước đó thì handover_lag là 1 hay 0 thì nghĩa là vẫn chọn cho UE đó quyết định phân bổ và check quyết định đó
        # =========================================================
        # CASE 3: handover_flag = 0 or 1 và UE chưa có mapping trước đó => đây là initial attachment (UE bị reject trước đó hoặc UE mới vào mạng)
        # =========================================================
        else:
            valid = self.check_action_valid(UE_idx, handover_flag, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)
            if not valid:
                is_pingpong = 0
                is_handover = 0
                reward = 0
                self.UE_manager.update_UE_request(UE_idx, {
                    "status": {"active": 0, "served": False, "reason": "Invalid action, choice RU, DU, CU other"},
                    "allocation" : self.UE_manager.empty_alloc(),
                    "prev_allocation": prev_alloc,
                    "pingpong": is_pingpong,
                    "handover_count": is_handover,
                })

                done = self.check_done()
                
                return self.get_state(), reward, done, {"success": False, "reason": "Invalid action, choice RU, DU, CU other"}
            
            
            # chưa được mapping trước đó nghĩa là không có phân bổ gì cho UE này trước đó
            is_handover = 0
            is_pingpong = 0
            # UE này chưa được mapping trước đó nên không cần release tài nguyên

            # radio_metrics = self.build_radio_metrics(
            #     UE_id=UE_idx,
            #     UE=UE,
            #     RU_choice=RU_choice,
            #     num_RB_alloc=num_RB_alloc,
            #     power_alloc=power_alloc,
            # )
            # Check xem việc quyết định mapping và phân bổ tài nguyên có đảm bảo các điều kiện ràng buộc
            feasible, msg = self.check_feasible(UE, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)
            self.write_radio_metrics_csv(
                UE_id=UE_idx,
                UE=UE,
                RU_choice=RU_choice,
                num_RB_alloc=num_RB_alloc,
                power_alloc=power_alloc,
                success=feasible,
                reason=msg,
            )
            # self.write_radio_metrics(
            #     metrics=radio_metrics,
            #     success=feasible,
            #     reason=msg,
            # )

            self.radio_debug_step += 1
            # Không thoả mãn
            if not feasible:
                reward = 0
                self.UE_manager.update_UE_request(UE_idx, {
                    "status": {"active": 0, "served": False, "reason": "mapping_not_valid"},
                    "allocation" : self.UE_manager.empty_alloc(),
                    "prev_allocation": prev_alloc,
                    "pingpong": is_pingpong,
                    "handover_count": is_handover,
                })

                done = self.check_done()
                
                return self.get_state(), reward, done, {"success": False, "reason": msg}
                        
            # Thoả mãn
            else:
                throughput_UE, latency_UE, cpu_DU_req, cpu_CU_req = msg[0], msg[1], msg[2], msg[3]
                # Mạng RAN cần cấp tài nguyên cho UE đó
                self.resource_manager.update_resources(RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc, cpu_DU_req, cpu_CU_req)

                # Cập nhật trạng thái cho UE
                self.UE_manager.update_UE_request(UE_idx, {
                    "status": {"active": 0, "served": True, "reason": "accepted_success"},
                    "allocation": {
                        "RU": RU_choice,
                        "DU": DU_choice,
                        "CU": CU_choice,
                        "num_RB_alloc": num_RB_alloc,
                        "power_alloc": power_alloc,
                        "throughput_bps": throughput_UE,
                        "delay_s": latency_UE,
                        "cpu_DU_req": cpu_DU_req,
                        "cpu_CU_req": cpu_CU_req,
                    },
                    "prev_allocation": prev_alloc,
                    "pingpong": is_pingpong,
                    "handover_count": is_handover,
                })

                # Tính toán reward từ việc phân bổ
                reward, acc_term, thr_term, lat_term, handover_term = self.compute_reward(UE, throughput_UE, latency_UE, is_handover, is_pingpong)

                done = self.check_done()
                        
                return self.get_state(), reward, done, {
                    "success": True,
                    "reason": "UE accept success: keep mapping",
                    "throughput_UE": throughput_UE,
                    "latency_UE": latency_UE,
                    "handover": is_handover,
                    "pingpong": is_pingpong,
                    "acc_term": acc_term,
                    "thr_term": thr_term,
                    "lat_term": lat_term,
                    "handover_term": handover_term,
                }
            

    # ======================================================================
    # Termination
    # ======================================================================
    def check_done(self):
        """Episode ends if no pending UE remains or any global resource depleted."""
        if self.UE_manager.check_UE_all_inactive():
            return True
        return False
    
    def close(self):
        if self.radio_log_file is not None:
            self.radio_log_file.flush()
            self.radio_log_file.close()
            self.radio_log_file = None