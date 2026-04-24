import numpy as np, copy
# Local modules - must exist
from numpy.linalg import norm
import networkx as nx

from config import *

class UEManager:
    def __init__(self, coordinates_RU):
        self.radius_in = 10
        self.radius_out = 1000
        self.SLICE_PRESET = dict(SLICE_PRESET)
        self.slice_names = list(self.SLICE_PRESET.keys())
        self.UE_requests = {}
        self.coordinates_RU = coordinates_RU


    def empty_status(self):
        return {"active": 1, "served": False, "reason": None}
    
    def empty_alloc(self):
        return {
            "RU": None,
            "DU": None,
            "CU": None,
            "num_RB_alloc": 0,
            "power_alloc": 0.0,
            "throughput_bps": 0.0,
            "delay_s": 0.0,
            "cpu_DU_req": 0.0,
            "cpu_CU_req": 0.0,
        }

    def calculate_distances(self, coordinate_UE):
        distances_RU_UE = []
        x_UE, y_UE = coordinate_UE
        for (x_RU, y_RU) in self.coordinates_RU:
            d = np.sqrt((x_RU - x_UE)**2 + (y_RU - y_UE)**2)
            distances_RU_UE.append(d)
        return distances_RU_UE


    def calculate_gain(self, distances_RU_UE):
        """
        distances_RU_UE: ma trận [num_RUs x num_UEs] khoảng cách RU-UE (m)
        bandwidth_per_RB: băng thông 1 RB (Hz)
        """
        # ------------------- Antenna config -------------------
        num_antennas = 32  # anten mỗi RU
        
        # ------------------- Noise power ----------------------
        k_B = 1.38064852e-23   # Boltzmann constant (J/K)
        T_K = 290              # Nhiệt độ (K)
        N0_W_per_Hz = k_B * T_K
        noise_figure_dB = 5
        noise_figure_linear = 10 ** (noise_figure_dB / 10)
        noise_power_RB = N0_W_per_Hz * bandwidth_per_RB * noise_figure_linear
        
        # ------------------- Carrier frequency ----------------
        f_c_GHz = 6
        gain = []
        for d in distances_RU_UE:
            # ------------------- Pathloss model (3GPP UMa) --------
            path_loss_db = 28 + 20 * np.log10(f_c_GHz) + 22 * np.log10(d)

            # ------------------- Pathloss linear ------------------
            path_loss_linear = 10 ** (-path_loss_db / 10)


            # kênh MIMO Rayleigh (num_antennas anten)
            h_real = np.random.randn(num_antennas)
            h_imag = np.random.randn(num_antennas)
            h = np.sqrt(path_loss_linear) * (h_real + 1j*h_imag) / np.sqrt(2)
                
            # power gain (chuẩn hóa theo norm-2)
            channel = norm(h, 2) ** 2

            gain.append(channel / noise_power_RB)
        return gain

    def compute_UE_metrics(self, coordinate_UE):
        distances = self.calculate_distances(coordinate_UE)
        gain = self.calculate_gain(distances)
        return distances, gain

    def set_coordinate_UE(self):
        angles = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(self.radius_in, self.radius_out)

        x = r * np.cos(angles)
        y = r * np.sin(angles)

        new_coords = (x, y)

        return new_coords
    
    def is_valid_position(self, coord):
        x, y = coord
        dist = np.hypot(x, y)
        return self.radius_in <= dist <= self.radius_out


    def build_UE_request(self, ue_id):
        slice_name = np.random.choice(self.slice_names, p=[0.7, 0.3])
        coordinate_UE = self.set_coordinate_UE()
        distances_RU_UE = self.calculate_distances(coordinate_UE)
        gains_UE = self.calculate_gain(distances_RU_UE)

        self.UE_requests[ue_id] = {
            "id": int(ue_id),
            **copy.deepcopy(self.SLICE_PRESET[slice_name]),

            "coordinate": coordinate_UE,
            "distances_RU_UE": distances_RU_UE,
            "gain": gains_UE,

            "status": self.empty_status(),

            "allocation": self.empty_alloc(),
            "prev_allocation": self.empty_alloc(),
            
            "pingpong": 0,
            "handover_count": 0,
        }
    
    
    def add_UEs_requests(self, new_UEs_request):
        if new_UEs_request <= 0:
            return []
        start_id = max(self.UE_requests.keys(), default=-1) + 1

        new_ids = list(range(start_id, start_id + new_UEs_request))

        for ue_id in new_ids:
            self.build_UE_request(ue_id)

        return new_ids

    def remove_UE(self, ue_id):
        return self.UE_requests.pop(ue_id, None)

    def remove_random_UEs(self, n_UEs_remove):
        ue_ids = list(self.UE_requests.keys())

        n_UEs_remove = min(int(n_UEs_remove), len(ue_ids))

        remove_ids = list(np.random.choice(ue_ids, n_UEs_remove, replace=False))

        removed_ues = []
        for ue_id in remove_ids:
            ue_info = self.remove_UE(ue_id)
            removed_ues.append((ue_id, ue_info))

        return removed_ues

    # def adjust_coordinates_UE(self, coordinate_UE): 
        
    #     delta_coordinate = 500
    #     x, y = coordinate_UE

    #     delta_x = np.random.uniform(-delta_coordinate, delta_coordinate)
    #     delta_y = np.random.uniform(-delta_coordinate, delta_coordinate)
            
    #     #Tọa độ mới sau khi thêm độ lệch
    #     new_x = x + delta_x
    #     new_y = y + delta_y
            
    #     return (new_x, new_y)
    
    def UE_mobility(self):
        #print("UE mobilityyyyyyyyyyyyyyyyyyyyyyy")
        max_UE = 100
        min_UE = 10
        max_UE_add = 2
        max_UE_departure = 2
        removed_ues_with_info = []
        num_UE_request = len(self.UE_requests)

        # =========================
        if num_UE_request <= min_UE:
            # Thêm UE mới
            id_new_UE = self.add_UEs_requests(max_UE_add)

            # UE còn lại di chuyển
            for UE_id in list(self.UE_requests.keys()):
                old_coordinate = self.UE_requests[UE_id]["coordinate"]
                new_coord = self.adjust_coordinates_UE(old_coordinate)

                if not self.is_valid_position(new_coord):
                    #print("Lỗi toạ độ mới")
                    ue_info = self.remove_UE(UE_id)
                    removed_ues_with_info.append((UE_id, ue_info))
                    continue

                # Update nếu hợp lệ
                distances, gain = self.compute_UE_metrics(new_coord)
                self.UE_requests[UE_id]["coordinate"] = new_coord
                self.UE_requests[UE_id]["distances_RU_UE"] = distances
                self.UE_requests[UE_id]["gain"] = gain
                self.UE_requests[UE_id]["status"]["active"] = 1
                

            return removed_ues_with_info, id_new_UE
        else: 
            # UE cũ rời đi ngẫu nhiên
            num_departures = np.random.randint(0, max_UE_departure + 1)

            departed_ues = self.remove_random_UEs(num_departures)
            removed_ues_with_info.extend(departed_ues)

            # UE cũ còn lại di chuyển
            for UE_id in list(self.UE_requests.keys()):
                old_coordinate = self.UE_requests[UE_id]["coordinate"]
                new_coord = self.adjust_coordinates_UE(old_coordinate)

                # if not self.is_valid_position(new_coord):
                #     #print("Lỗi toạ độ mới")
                #     ue_info = self.remove_UE(UE_id)
                #     removed_ues_with_info.append((UE_id, ue_info))
                #     continue

                
                 # Update nếu hợp lệ
                distances, gain = self.compute_UE_metrics(new_coord)
                self.UE_requests[UE_id]["coordinate"] = new_coord
                self.UE_requests[UE_id]["distances_RU_UE"] = distances
                self.UE_requests[UE_id]["gain"] = gain
                self.UE_requests[UE_id]["status"]["active"] = 1
            
            # UE mới thêm vào
            new_UEs_request = np.random.randint(0, min(max_UE_add, max_UE - num_UE_request) + 1)
            id_new_UE = self.add_UEs_requests(new_UEs_request)

            return removed_ues_with_info, id_new_UE

    def adjust_coordinates_UE(self, coordinate_UE):
        delta_coordinate = 1000
        x, y = coordinate_UE

        delta_x = np.random.uniform(-delta_coordinate, delta_coordinate)
        delta_y = np.random.uniform(-delta_coordinate, delta_coordinate)

        new_x = x + delta_x
        new_y = y + delta_y

        # Đưa UE về đúng vùng mô phỏng nếu đi ra ngoài
        dist = np.hypot(new_x, new_y)

        if dist < self.radius_in:
            if dist < 1e-12:
                angle = np.random.uniform(0, 2 * np.pi)
                new_x = self.radius_in * np.cos(angle)
                new_y = self.radius_in * np.sin(angle)
            else:
                scale = self.radius_in / dist
                new_x *= scale
                new_y *= scale

        elif dist > self.radius_out:
            scale = self.radius_out / dist
            new_x *= scale
            new_y *= scale

        return (new_x, new_y)


    # def UE_mobility(self):
    #     """
    #     Chỉ di chuyển UE, không thêm và không xóa UE.
    #     Trả về:
    #         removed_ues_with_info = []   # luôn rỗng
    #         id_new_UE = []               # luôn rỗng
    #     """
    #     removed_ues_with_info = []
    #     id_new_UE = []

    #     for UE_id in list(self.UE_requests.keys()):
    #         old_coordinate = self.UE_requests[UE_id]["coordinate"]
    #         new_coord = self.adjust_coordinates_UE(old_coordinate)

    #         distances, gain = self.compute_UE_metrics(new_coord)
    #         self.UE_requests[UE_id]["coordinate"] = new_coord
    #         self.UE_requests[UE_id]["distances_RU_UE"] = distances
    #         self.UE_requests[UE_id]["gain"] = gain
    #         self.UE_requests[UE_id]["status"]["active"] = 1

    #     return removed_ues_with_info, id_new_UE

    def update_UE_request(self, ue_id, update_dict):
        ue = self.UE_requests[ue_id]

        for key, value in update_dict.items():
            # Nếu là dict → update sâu (merge)
            if isinstance(value, dict) and key in ue and isinstance(ue[key], dict):
                ue[key].update(value)
            else:
                # Gán trực tiếp
                ue[key] = value

    def check_UE_all_inactive(self):
        if not self.UE_requests:
            return True
        return all(int(ue["status"].get("active", 0)) == 0 for ue in self.UE_requests.values())



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
    def __init__(self, num_UEs, num_RBs, total_nodes, num_RUs, num_DUs, num_CUs):

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
        self.UE_manager = UEManager(self.resource_manager.coordinates_RU)
        self.resource_manager.reset()
        _ = self.UE_manager.add_UEs_requests(self.num_UEs)
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
        eps = 1e-12
        c_speed = 3e8  # tốc độ ánh sáng (m/s)
        #xhaul_delay_s = 1e-3  # độ trễ xhaul (s)

        latency_path = {}
        mu_du = cpu_DU_req / cycles_per_packet
        mu_cu = cpu_CU_req / cycles_per_packet
        rho_j = lambda_default_pps / mu_du
        # -----------------------------
        # 1) Propagation latency
        # -----------------------------
        #L_prop = ((distances_RU_UE / c_speed) + xhaul_delay_s) * 1.0  # Phi_i 
        L_prop = ((distances_RU_UE / c_speed)) * 1.0  # Phi_i 

        # -----------------------------
        # 2) Transmission latency
        # -----------------------------
        L_trans = packet_size_bits / throughput_bps

        # -----------------------------
        # 3) Queuing latency
        # -----------------------------
        L_queu = rho_j / (mu_du - lambda_default_pps)
        
        # -----------------------------
        # 4) Processing latency
        # -----------------------------
        L_proc_du = 1 / (mu_du - lambda_default_pps)
        L_proc_cu = 1 / (mu_cu - lambda_default_pps)
        # -----------------------------
        # 5) Total E2E latency
        # -----------------------------
        total_latency = L_prop + L_trans + L_queu + L_proc_du + L_proc_cu

        latency_path= {
            "prop": L_prop,
            "trans": L_trans,
            "queu": L_queu,
            "proc_du": L_proc_du,
            "proc_cu": L_proc_cu,
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

        if RB < 0 or RB > self.resource_manager.max_RBs_per_UE:
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
                    feasible, msg = self.check_feasible(UE, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)
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
                    feasible, msg = self.check_feasible(UE, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)

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

            # Check xem việc quyết định mapping và phân bổ tài nguyên có đảm bảo các điều kiện ràng buộc
            feasible, msg = self.check_feasible(UE, RU_choice, DU_choice, CU_choice, num_RB_alloc, power_alloc)

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
