import numpy as np
import cv2
import random
import pickle
import os
import time

# ==================== 配置参数 ====================
DT = 0.1
BASE_THRESHOLD = 1.0
RESET_POTENTIAL = 0.0

A_PLUS = 0.05
A_MINUS = 0.06
TAU_STDP = 20.0

ERROR_STDP_RATE = 0.01          # 误差对权重的调整系数（双向）
THRESHOLD_ADJUST_RATE = 0.001

SPACE_SIZE = 10.0
FACE_Z_POS = 15.0
FACE_Z_NEG = -15.0
FACE_RADIUS = 8.0

ALIGNMENT_THRESHOLD = 0.98
NEW_SYNAPSE_WEIGHT = 0.5

IMG_WIDTH = 64
IMG_HEIGHT = 64
NUM_PIXELS = IMG_WIDTH * IMG_HEIGHT
NUM_RGB_VALUES = NUM_PIXELS * 3

NUM_STARS = 1280

CONN_PER_FACE_MIN = 1
CONN_PER_FACE_MAX = 3

MODEL_SAVE_PATH = "universe_snn_model.pkl"
SAVE_INTERVAL = 20
MAX_SPIKING_FOR_GROWTH = 80

MODE_TRAIN = 'train'
MODE_INFER = 'infer'

# ===== 新增：剪枝间隔（帧数） =====
PRUNE_INTERVAL = 10
PRUNE_THRESHOLD = 1e-9          # 权重小于此值视为零

# ==================== 核心 SNN 模型 ====================
class UniverseSNN:
    def __init__(self, num_face_neurons: int, num_stars: int = NUM_STARS, mode: str = MODE_TRAIN):
        self.num_face = num_face_neurons
        self.num_stars = num_stars
        self.num_total = num_face_neurons * 2 + num_stars
        self.time = 0.0
        self.mode = mode

        self.potentials = np.zeros(self.num_total, dtype=np.float64)
        self.input_currents = np.zeros(self.num_total, dtype=np.float64)
        self.last_spike_time = np.full(self.num_total, -1000.0)
        self.thresholds = np.full(self.num_total, BASE_THRESHOLD, dtype=np.float64)

        self.positions = np.zeros((self.num_total, 3), dtype=np.float64)
        self._init_positions()

        self.syn_pre = np.array([], dtype=np.int64)
        self.syn_post = np.array([], dtype=np.int64)
        self.syn_weight = np.array([], dtype=np.float64)
        self.pre_to_syn: dict[int, list[int]] = {}

        self.spiked_prev: set[int] = set()
        self.spiked_curr: set[int] = set()

        self._init_connections()

    def _init_positions(self):
        nf, ns = self.num_face, self.num_stars
        for i in range(nf):
            r = FACE_RADIUS * np.sqrt(np.random.random())
            th = np.random.random() * 2 * np.pi
            self.positions[i] = [r * np.cos(th), r * np.sin(th), FACE_Z_NEG]
        for i in range(nf):
            r = FACE_RADIUS * np.sqrt(np.random.random())
            th = np.random.random() * 2 * np.pi
            self.positions[nf + i] = [r * np.cos(th), r * np.sin(th), FACE_Z_POS]
        for i in range(ns):
            phi = np.random.uniform(0, np.pi)
            th = np.random.uniform(0, 2 * np.pi)
            r = SPACE_SIZE * np.cbrt(np.random.uniform())
            idx = 2 * nf + i
            self.positions[idx] = [
                r * np.sin(phi) * np.cos(th),
                r * np.sin(phi) * np.sin(th),
                r * np.cos(phi),
            ]

    def _init_connections(self):
        nf, ns = self.num_face, self.num_stars
        star_start = 2 * nf
        star_ids = list(range(star_start, star_start + ns))
        pre_list, post_list, w_list = [], [], []
        for i in range(nf):
            n_conn = random.randint(CONN_PER_FACE_MIN, CONN_PER_FACE_MAX)
            for t in random.sample(star_ids, min(n_conn, ns)):
                pre_list.append(i); post_list.append(t); w_list.append(0.1)
        for i in range(nf):
            face_idx = nf + i
            n_conn = random.randint(CONN_PER_FACE_MIN, CONN_PER_FACE_MAX)
            for t in random.sample(star_ids, min(n_conn, ns)):
                pre_list.append(face_idx); post_list.append(t); w_list.append(0.1)
        for s in star_ids:
            n_conn = random.randint(1, 3)
            for t in random.sample(star_ids, min(n_conn + 1, ns)):
                if t != s:
                    pre_list.append(s); post_list.append(t); w_list.append(0.5)
        self.syn_pre = np.array(pre_list, dtype=np.int64)
        self.syn_post = np.array(post_list, dtype=np.int64)
        self.syn_weight = np.array(w_list, dtype=np.float64)
        self._build_pre_index()

    def _build_pre_index(self):
        self.pre_to_syn = {}
        for idx in range(len(self.syn_pre)):
            p = int(self.syn_pre[idx])
            if p not in self.pre_to_syn:
                self.pre_to_syn[p] = []
            self.pre_to_syn[p].append(idx)

    def step(self, error: np.ndarray = None):
        self.time += DT

        # 前向突触电流
        synaptic_currents = np.zeros(self.num_total, dtype=np.float64)
        if self.spiked_prev:
            spiked_prev_bool = np.zeros(self.num_total, dtype=bool)
            spiked_prev_bool[list(self.spiked_prev)] = True
            active_mask = spiked_prev_bool[self.syn_pre]
            np.add.at(
                synaptic_currents,
                self.syn_post[active_mask],
                self.syn_weight[active_mask],
            )

        self.potentials += (self.input_currents + synaptic_currents) * DT
        self.input_currents[:] = 0.0

        # ===== 安全钳制（防止溢出，范围很宽不影响学习）=====
        np.clip(self.potentials, -50.0, 50.0, out=self.potentials)

        # 发放脉冲
        spike_mask = self.potentials >= self.thresholds
        spiked_ids = np.where(spike_mask)[0]
        self.potentials[spike_mask] = RESET_POTENTIAL
        self.last_spike_time[spiked_ids] = self.time
        self.spiked_curr = set(spiked_ids.tolist())

        # 正向STDP
        self._apply_stdp()

        # 反向误差STDP + 阈值调整
        if error is not None:
            self._apply_error_stdp(error)
            self._adjust_thresholds(error)

        # 结构可塑性（仅训练）
        if self.mode == MODE_TRAIN:
            self._apply_plasticine_growth()

        self.spiked_prev = self.spiked_curr

    def _apply_stdp(self):
        if not self.spiked_prev or not self.spiked_curr:
            return
        prev_bool = np.zeros(self.num_total, dtype=bool)
        prev_bool[list(self.spiked_prev)] = True
        curr_bool = np.zeros(self.num_total, dtype=bool)
        curr_bool[list(self.spiked_curr)] = True
        ltp = prev_bool[self.syn_pre] & curr_bool[self.syn_post]
        if np.any(ltp):
            self.syn_weight[ltp] += A_PLUS * np.exp(-DT / TAU_STDP)
        ltd = prev_bool[self.syn_post] & curr_bool[self.syn_pre]
        if np.any(ltd):
            self.syn_weight[ltd] -= A_MINUS * np.exp(-DT / TAU_STDP)
        np.clip(self.syn_weight, 0.0, 2.0, out=self.syn_weight)

    def _apply_error_stdp(self, error):
        """
        双向反向STDP：根据误差符号和大小调整权重
        error > 0 表示预测偏低，增强权重；error < 0 预测偏高，削弱权重
        """
        if len(self.spiked_curr) == 0:
            return
        curr_bool = np.zeros(self.num_total, dtype=bool)
        curr_bool[list(self.spiked_curr)] = True
        mask_pre = curr_bool[self.syn_pre]
        mask_post = curr_bool[self.syn_post]
        active_synapses = mask_pre | mask_post

        face2_start = self.num_face
        face2_end = 2 * self.num_face
        post_is_face2 = (self.syn_post >= face2_start) & (self.syn_post < face2_end)
        active_face2 = active_synapses & post_is_face2

        if not np.any(active_face2):
            return

        post_ids = self.syn_post[active_face2]
        post_local = post_ids - face2_start
        error_selected = error[post_local]   # 符号和大小
        # 权重更新：与误差成正比，正向误差增强，负向削弱
        delta = ERROR_STDP_RATE * error_selected * self.syn_weight[active_face2]
        self.syn_weight[active_face2] += delta
        np.clip(self.syn_weight, 0.0, 2.0, out=self.syn_weight)

    def _adjust_thresholds(self, error):
        """根据误差调整Face2阈值：预测偏低则降低阈值，反之升高"""
        face2_start = self.num_face
        face2_end = 2 * self.num_face
        for i in range(self.num_face):
            idx = face2_start + i
            delta = -THRESHOLD_ADJUST_RATE * error[i]   # error正（预测低）=> delta负=>阈值降
            self.thresholds[idx] += delta
            self.thresholds[idx] = np.clip(self.thresholds[idx], 0.1, 10.0)

    def _apply_plasticine_growth(self):
        if len(self.spiked_curr) < 2:
            return
        spiking = list(self.spiked_curr)
        if len(spiking) > MAX_SPIKING_FOR_GROWTH:
            spiking = random.sample(spiking, MAX_SPIKING_FOR_GROWTH)
        existing = set(zip(self.syn_pre.tolist(), self.syn_post.tolist()))
        for i in range(len(spiking)):
            for j in range(i + 1, len(spiking)):
                src, dst = spiking[i], spiking[j]
                if (src, dst) in existing:
                    continue
                vec = self.positions[dst] - self.positions[src]
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    continue
                direction = vec / dist
                best_relay, best_d = None, float("inf")
                for k in range(len(spiking)):
                    if k == i or k == j:
                        continue
                    mid = spiking[k]
                    v2 = self.positions[mid] - self.positions[src]
                    d2 = np.linalg.norm(v2)
                    if d2 < 1e-9 or d2 >= dist:
                        continue
                    if np.dot(direction, v2 / d2) > ALIGNMENT_THRESHOLD and d2 < best_d:
                        best_d, best_relay = d2, mid
                if best_relay is not None:
                    if (src, best_relay) not in existing:
                        self._add_synapse(src, best_relay, NEW_SYNAPSE_WEIGHT)
                        existing.add((src, best_relay))
                    if (best_relay, dst) not in existing:
                        self._add_synapse(best_relay, dst, NEW_SYNAPSE_WEIGHT)
                        existing.add((best_relay, dst))
                else:
                    self._add_synapse(src, dst, NEW_SYNAPSE_WEIGHT)
                    existing.add((src, dst))

    def _add_synapse(self, pre: int, post: int, w: float):
        idx = len(self.syn_pre)
        self.syn_pre = np.append(self.syn_pre, pre)
        self.syn_post = np.append(self.syn_post, post)
        self.syn_weight = np.append(self.syn_weight, w)
        self.pre_to_syn.setdefault(int(pre), []).append(idx)

    # ===== 新增：剪除权重为零的连接 =====
    def prune_zero_weights(self, threshold: float = PRUNE_THRESHOLD):
        """
        删除所有权重 <= threshold 的突触，并重建 pre_to_syn 索引。
        """
        keep_mask = self.syn_weight > threshold
        num_before = len(self.syn_pre)
        if not np.all(keep_mask):
            self.syn_pre = self.syn_pre[keep_mask]
            self.syn_post = self.syn_post[keep_mask]
            self.syn_weight = self.syn_weight[keep_mask]
            self._build_pre_index()
            removed = num_before - len(self.syn_pre)
            print(f"[剪枝] 删除了 {removed} 个零权重连接，剩余 {len(self.syn_pre)}")
        else:
            print("[剪枝] 无零权重连接需删除")

    def input_face_1(self, signals: np.ndarray):
        self.input_currents[:self.num_face] += signals

    def input_face_2(self, signals: np.ndarray):
        self.input_currents[self.num_face:2*self.num_face] += signals

    def get_output_face_2(self) -> np.ndarray:
        return self.potentials[self.num_face:2*self.num_face].copy()

    def set_mode(self, mode: str):
        if mode in (MODE_TRAIN, MODE_INFER):
            self.mode = mode
            print(f"切换模式为: {mode}")
        else:
            raise ValueError("模式必须是 'train' 或 'infer'")

    def save_model(self, path: str = MODEL_SAVE_PATH):
        data = {
            "time": self.time,
            "num_face": self.num_face,
            "num_stars": self.num_stars,
            "mode": self.mode,
            "potentials": self.potentials,
            "thresholds": self.thresholds,
            "last_spike_time": self.last_spike_time,
            "positions": self.positions,
            "syn_pre": self.syn_pre,
            "syn_post": self.syn_post,
            "syn_weight": self.syn_weight,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[保存] 模型 → {path}  (连接数={len(self.syn_pre)})")

    @classmethod
    def load_model(cls, path: str = MODEL_SAVE_PATH) -> "UniverseSNN":
        with open(path, "rb") as f:
            data = pickle.load(f)
        snn = cls.__new__(cls)
        snn.num_face = data["num_face"]
        snn.num_stars = data["num_stars"]
        snn.num_total = snn.num_face * 2 + snn.num_stars
        snn.time = data["time"]
        snn.mode = data.get("mode", MODE_TRAIN)
        snn.potentials = data["potentials"]
        snn.input_currents = np.zeros(snn.num_total, dtype=np.float64)
        snn.last_spike_time = data["last_spike_time"]
        snn.positions = data["positions"]
        snn.syn_pre = data["syn_pre"]
        snn.syn_post = data["syn_post"]
        snn.syn_weight = data["syn_weight"]
        snn.thresholds = data.get("thresholds", np.full(snn.num_total, BASE_THRESHOLD))
        snn.spiked_prev = set()
        snn.spiked_curr = set()
        snn._build_pre_index()
        print(f"[加载] 模型 ← {path}  (连接数={len(snn.syn_pre)})")
        return snn

# ==================== 摄像头采集与显示 ====================
def grab_rgb_flat(cap) -> np.ndarray | None:
    ret, frame = cap.read()
    if not ret:
        return None
    frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb.reshape(-1).astype(np.float64) / 255.0

def face2_output_to_image(face2_potentials: np.ndarray) -> np.ndarray:
    arr = np.clip(face2_potentials, 0.0, 1.0)
    img_rgb = (arr * 255).astype(np.uint8).reshape(IMG_HEIGHT, IMG_WIDTH, 3)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr

# ==================== 主循环 ====================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    if os.path.exists(MODEL_SAVE_PATH):
        print("检测到已有模型，正在加载 …")
        snn = UniverseSNN.load_model(MODEL_SAVE_PATH)
    else:
        print("未检测到模型文件，正在初始化 …")
        t0 = time.time()
        snn = UniverseSNN(num_face_neurons=NUM_RGB_VALUES, num_stars=NUM_STARS, mode=MODE_TRAIN)
        print(f"初始化完成，耗时 {time.time() - t0:.1f}s")
        print(f"  面神经元数  : {NUM_RGB_VALUES} × 2 面")
        print(f"  中间层神经元: {NUM_STARS}")
        print(f"  初始连接数  : {len(snn.syn_pre)}")

    print("\n操作: q=退出  s=手动保存模型  t=切换训练/推理模式")
    print("-" * 50)
    cv2.namedWindow("Face2 Output", cv2.WINDOW_NORMAL)

    frame_count = 0
    face2_output_t = None

    while True:
        rgb_flat = grab_rgb_flat(cap)
        if rgb_flat is None:
            break

        if frame_count == 0:
            snn.input_face_1(rgb_flat)
            snn.step()          # 无误差
            face2_output_t = snn.get_output_face_2()
        else:
            # ===== 误差为 真实 - 预测 =====
            error = rgb_flat - face2_output_t   # 预测偏低时为正
            snn.input_face_1(rgb_flat)
            snn.input_face_2(error)            # 注入误差电流
            snn.step(error=error)
            face2_output_t = snn.get_output_face_2()

        # ===== 训练模式下定期剪枝 =====
        if snn.mode == MODE_TRAIN and frame_count % PRUNE_INTERVAL == 0:
            snn.prune_zero_weights()

        face2_img = face2_output_to_image(face2_output_t)
        cv2.imshow("Face2 Output", face2_img)

        frame_count += 1
        if frame_count % SAVE_INTERVAL == 0:
            snn.save_model()

        f2 = face2_output_t
        print(
            f"Frame {frame_count:>5d} | "
            f"模式 {snn.mode:>5s} | "
            f"连接 {len(snn.syn_pre):>8d} | "
            f"放电 {len(snn.spiked_curr):>6d} | "
            f"Face2 [{f2.min():+.3f}, {f2.max():+.3f}] | "
            f"阈值 [{snn.thresholds[snn.num_face:2*snn.num_face].min():+.2f}, "
            f"{snn.thresholds[snn.num_face:2*snn.num_face].max():+.2f}]"
        )

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            snn.save_model()
        elif key == ord('t'):
            new_mode = MODE_INFER if snn.mode == MODE_TRAIN else MODE_TRAIN
            snn.set_mode(new_mode)

    snn.save_model()
    cap.release()
    cv2.destroyAllWindows()
    print("程序结束。")

if __name__ == "__main__":
    main()
