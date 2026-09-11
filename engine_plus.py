"""
SUSNN 核心脉冲引擎 —— 纯动作输出版
============================================================

对应设计文档《SUSNN: Self-Unifying Spiking Neural Network》：

- 预全连接初始化：连接半径内的所有神经元对**全部**建边，权重取小随机值。
- 三个功能区：
    · 第一面：接收外部输入（face_rows × face_cols 网格，z=0）
    · 中间层：坐标由外部星系数据提供
    · 第二面：仅由动作神经元组成（z=space_depth），不对外输出
- 无睡眠态、无结构可塑性、无误差驱动、无预测闭环。
- 权重**仅由 STDP 调整**，无额外衰减。反向时序会被 LTD 压向 0，等效剪枝。

对外接口：
    inject_input()           向第一面注入 [-1,1] 强度图
    read_action_spikes()     读动作神经元本轮发放
    read_action_potentials() 读动作神经元当前膜电位
    get_stats() / get_layer_stats()
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


# ===========================================================================
# 第零部分：CuPy 惰性加载与通用工具
# ===========================================================================

_CUPY_CACHE: Dict[str, object] = {}


def _try_load_cupy():
    if "cp" in _CUPY_CACHE:
        return _CUPY_CACHE["cp"]
    cp = None
    try:
        import cupy as _cp
        if _cp.cuda.runtime.getDeviceCount() >= 1:
            cp = _cp
    except Exception:
        cp = None
    _CUPY_CACHE["cp"] = cp
    return cp


def _block_cross(starts: np.ndarray, counts: np.ndarray,
                 u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """批量单元格叉积：第 k 组 = u[k] 单元格成员 × v[k] 单元格成员。"""
    ca = counts[u]
    cb = counts[v]
    sizes = ca * cb
    total = int(sizes.sum())
    if total <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    blk = np.empty(sizes.size, np.int64)
    blk[0] = 0
    np.cumsum(sizes[:-1], out=blk[1:])
    within = np.arange(total, dtype=np.int64) - np.repeat(blk, sizes)
    cb_rep = np.repeat(cb, sizes)
    ia = np.repeat(starts[u], sizes) + within // cb_rep
    ib = np.repeat(starts[v], sizes) + within % cb_rep
    return ia, ib


# ===========================================================================
# 第一部分：核心脉冲引擎
# ===========================================================================

class PulseEngine:
    """
    核心脉冲引擎（向量化实现）。
    第一面 = 输入，中间层 = 星系坐标，第二面 = 纯动作神经元。
    """

    LAYER_FIRST  = 0
    LAYER_MIDDLE = 1
    LAYER_ACTION = 2

    _STATE_ARRAYS = ("membrane", "thresholds", "traces", "roll_sum", "fired",
                     "pot_history", "edge_pre", "edge_post", "edge_w")

    # -------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------

    def __init__(self,
                 face_rows: int = 32,
                 face_cols: int = 32,
                 n_middle: int = 1000,
                 n_action: int = 8,
                 connection_radius: float = 4.0,
                 space_depth: float = 16.0,
                 window_size: int = 200,
                 base_threshold: float = 1.0,
                 adaptation_rate: float = 0.3,
                 stdp_lr_plus: float = 0.01,
                 stdp_lr_minus: float = 0.012,
                 stdp_tau: float = 20.0,
                 w_min: float = 0.0,
                 w_max: float = 5.0,
                 init_weight_lo: float = 0.02,
                 init_weight_hi: float = 0.30,
                 galaxy_coords: Optional[np.ndarray] = None,
                 seed: int = 42,
                 device: str = "auto",
                 dtype=np.float64,
                 auto_bench_steps: int = 30):
        self.rng = np.random.default_rng(seed)

        # ---- 结构与动力学参数 ----
        self.face_rows       = face_rows
        self.face_cols       = face_cols
        self.n_face          = face_rows * face_cols
        self.n_middle        = n_middle
        self.n_action        = n_action
        self.n_total         = self.n_face + n_middle + n_action
        self.connection_radius = connection_radius
        self.space_depth     = space_depth
        self.window_size     = window_size
        self.base_threshold  = base_threshold
        self.adaptation_rate = adaptation_rate
        self.stdp_lr_plus    = stdp_lr_plus
        self.stdp_lr_minus   = stdp_lr_minus
        self.stdp_tau        = stdp_tau
        self.w_min           = w_min
        self.w_max           = w_max
        self.init_weight_lo  = init_weight_lo
        self.init_weight_hi  = init_weight_hi

        # ---- 设备 ----
        self.dtype       = np.dtype(dtype)
        self._xp         = np
        self.device_name = "cpu(numpy)"

        # ---- 构建 ----
        self._build_network(galaxy_coords)
        self._init_connections()
        self._init_state()
        self._trace_decay = float(np.exp(-1.0 / self.stdp_tau))

        if device in ("auto", "gpu"):
            self._select_device(device, auto_bench_steps)
        elif device != "cpu":
            raise ValueError(f"device 须为 'auto'/'cpu'/'gpu', 收到 {device!r}")

    # -------------------------------------------------------------------
    # 网络构建：第一面 + 中间层 + 动作面
    # -------------------------------------------------------------------

    def _build_network(self, galaxy_coords: Optional[np.ndarray]):
        """
        坐标布局：
          第一面  : z = 0,      网格 (face_cols × face_rows)
          中间层  : z ∈ [0.3, space_depth - 0.3]，坐标由外部缩放
          第二面  : z = space_depth，仅含 n_action 个动作神经元
        """
        positions: List[List[float]] = []
        layer_ids: List[int] = []
        idx = 0

        # ---- 第一面：接收输入 ----
        for r in range(self.face_rows):
            for c in range(self.face_cols):
                positions.append([float(c), float(r), 0.0])
                layer_ids.append(self.LAYER_FIRST)
        self.first_start, self.first_end = 0, idx + self.n_face
        idx = self.first_end

        # ---- 中间层：坐标来自外部 ----
        if self.n_middle > 0:
            if galaxy_coords is None:
                raise ValueError(
                    "n_middle > 0 时必须提供 galaxy_coords（中间层空间坐标）。")
            coords_in = np.asarray(galaxy_coords, dtype=np.float64)
            if coords_in.ndim != 2 or coords_in.shape[1] != 3:
                raise ValueError(
                    f"galaxy_coords 须为 (n, 3) 形状, 收到 {coords_in.shape}")
            if coords_in.shape[0] < self.n_middle:
                raise ValueError(
                    f"galaxy_coords 数量不足: 需要 {self.n_middle}, "
                    f"实际 {coords_in.shape[0]}")
            coords = self._scale_galaxy_coords(coords_in[:self.n_middle])
        else:
            coords = np.empty((0, 3), dtype=np.float64)

        for i in range(self.n_middle):
            positions.append(coords[i].tolist())
            layer_ids.append(self.LAYER_MIDDLE)
        self.middle_start, self.middle_end = idx, idx + self.n_middle
        idx = self.middle_end

        # ---- 第二面 = 动作面：只有动作神经元 ----
        # 沿 x 轴均匀铺在 z = space_depth 平面上，y 居中。
        # 动作神经元之间相距较远时互不连接；间距足够近则由半径决定。
        for i in range(self.n_action):
            if self.n_action > 1:
                x = (i + 0.5) * self.face_cols / self.n_action
            else:
                x = self.face_cols * 0.5
            positions.append([x, self.face_rows / 2.0, self.space_depth])
            layer_ids.append(self.LAYER_ACTION)
        self.second_start, self.second_end = idx, idx + self.n_action
        idx = self.second_end

        assert idx == self.n_total
        self.positions = np.array(positions, dtype=np.float32)
        self.layer_ids = np.array(layer_ids, dtype=np.int32)

        self.is_input_neuron  = (self.layer_ids == self.LAYER_FIRST)
        self.is_middle_neuron = (self.layer_ids == self.LAYER_MIDDLE)
        self.is_action_neuron = (self.layer_ids == self.LAYER_ACTION)

        # 便捷别名
        self.action_start, self.action_end = self.second_start, self.second_end

    def _scale_galaxy_coords(self, coords: np.ndarray) -> np.ndarray:
        """外部坐标只提供"相对形状"，引擎 min-max 归一化映射进网络空间。"""
        coords = coords.astype(np.float64).copy()
        ranges = [(0.0, float(self.face_cols)),
                  (0.0, float(self.face_rows)),
                  (0.3, self.space_depth - 0.3)]
        for dim in range(3):
            lo, hi = coords[:, dim].min(), coords[:, dim].max()
            t_lo, t_hi = ranges[dim]
            if hi > lo:
                coords[:, dim] = t_lo + (coords[:, dim] - lo) / (hi - lo) * (t_hi - t_lo)
            else:
                coords[:, dim] = (t_lo + t_hi) / 2.0
        return coords

    # -------------------------------------------------------------------
    # 空间近邻配对（网格法，向量化）
    # -------------------------------------------------------------------

    def _pairs_within_radius(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 ids 中所有空间距离 < connection_radius 的无序对 (lo < hi)。"""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size < 2:
            e = np.empty(0, np.int64)
            return e, e.copy()

        cs, r2 = self.connection_radius, self.connection_radius ** 2
        cell = np.floor(self.positions[ids] / cs).astype(np.int64)
        Mx = int(cell[:, 0].max()) + 1
        My = int(cell[:, 1].max()) + 1
        Mz = int(cell[:, 2].max()) + 1
        key = cell[:, 0] + Mx * (cell[:, 1] + My * cell[:, 2])

        order = np.argsort(key, kind="stable")
        key_s, ids_s = key[order], ids[order]
        uk, starts = np.unique(key_s, return_index=True)
        counts = np.diff(np.append(starts, key_s.size))
        ux, uy, uz = uk % Mx, (uk // Mx) % My, uk // (Mx * My)

        lo_list: List[np.ndarray] = []
        hi_list: List[np.ndarray] = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) <= (0, 0, 0):
                        continue
                    vx, vy, vz = ux + dx, uy + dy, uz + dz
                    ok = ((vx >= 0) & (vx < Mx) & (vy >= 0) & (vy < My)
                          & (vz >= 0) & (vz < Mz))
                    if not ok.any():
                        continue
                    src = np.flatnonzero(ok)
                    nk = vx[ok] + Mx * (vy[ok] + My * vz[ok])
                    p = np.searchsorted(uk, nk)
                    np.clip(p, 0, uk.size - 1, out=p)
                    hit = uk[p] == nk
                    if not hit.any():
                        continue
                    ia, ib = _block_cross(starts, counts, src[hit], p[hit])
                    if ia.size == 0:
                        continue
                    a, b = ids_s[ia], ids_s[ib]
                    d = self.positions[a] - self.positions[b]
                    keep = np.einsum("ij,ij->i", d, d) < r2
                    if keep.any():
                        lo_list.append(np.minimum(a[keep], b[keep]))
                        hi_list.append(np.maximum(a[keep], b[keep]))

        multi = np.flatnonzero(counts > 1)
        if multi.size:
            ia, ib = _block_cross(starts, counts, multi, multi)
            a, b = ids_s[ia], ids_s[ib]
            keep = a < b
            if keep.any():
                a, b = a[keep], b[keep]
                d = self.positions[a] - self.positions[b]
                keep2 = np.einsum("ij,ij->i", d, d) < r2
                if keep2.any():
                    lo_list.append(a[keep2])
                    hi_list.append(b[keep2])

        if not lo_list:
            e = np.empty(0, np.int64)
            return e, e.copy()
        return np.concatenate(lo_list), np.concatenate(hi_list)

    # -------------------------------------------------------------------
    # 连接初始化 —— 预全连接
    # -------------------------------------------------------------------

    def _init_connections(self):
        """所有满足连接半径的神经元对**全部**建边，权重取小随机值。"""
        ids = np.arange(self.n_total, dtype=np.int64)
        lo, hi = self._pairs_within_radius(ids)
        if lo.size:
            w = self.rng.uniform(self.init_weight_lo, self.init_weight_hi,
                                 size=lo.size).astype(self.dtype, copy=False)
        else:
            w = np.zeros(0, dtype=self.dtype)
        self.edge_pre  = lo.astype(np.int32)
        self.edge_post = hi.astype(np.int32)
        self.edge_w    = w

    # -------------------------------------------------------------------
    # 状态初始化
    # -------------------------------------------------------------------

    def _init_state(self):
        N, dt = self.n_total, self.dtype
        self.membrane   = np.zeros(N, dtype=dt)
        self.thresholds = np.full(N, self.base_threshold, dtype=dt)
        self.traces     = np.zeros(N, dtype=dt)
        self.pot_history = np.zeros((self.window_size, N), dtype=np.float32)
        self.roll_sum    = np.zeros(N, dtype=dt)
        self.hist_ptr    = 0
        self.pulse_buf   = [np.zeros(N, dtype=dt), np.zeros(N, dtype=dt)]
        self._buf        = 0
        self.fired       = np.zeros(N, dtype=bool)
        self.time_step   = 0

    # -------------------------------------------------------------------
    # 核心：一个时间步
    # -------------------------------------------------------------------

    def step(self):
        """
        一个时间步 = 所有神经元各更新一次（同步扫描语义）。

        1. 双缓冲交换
        2. STDP 迹衰减
        3. 到达脉冲叠加
        4. 阈值判定 → 发放 / 仅积分；发放者减法重置
        5. STDP：按迹更新权重（无额外衰减项）
        6. 发放神经元迹 +1
        7. 脉冲按（STDP 后）权重散射进下一时间步缓冲
        8. 滑动窗口滚动和 + 动态阈值
        """
        xp = self._xp
        N = self.n_total

        # 1. 双缓冲交换
        cur, nxt = self._buf, 1 - self._buf
        self._buf = nxt
        pulse_in, pulse_out = self.pulse_buf[cur], self.pulse_buf[nxt]

        # 2. STDP 迹衰减
        self.traces *= self._trace_decay

        # 3. 到达脉冲叠加
        self.membrane += pulse_in

        # 4. 阈值判定 + 减法重置
        fire = self.membrane >= self.thresholds
        fire_f = fire.astype(self.dtype)
        self.membrane -= self.thresholds * fire_f
        self.fired = fire

        # 5. STDP + 7. 脉冲散射
        if self.edge_pre.size:
            e_pre, e_post = self.edge_pre, self.edge_post
            fire_pre  = fire_f[e_pre]
            fire_post = fire_f[e_post]
            # 纯 STDP：前→后 LTP，后→前 LTD。无独立衰减。
            delta = (self.stdp_lr_plus  * self.traces[e_pre]  * fire_post
                     - self.stdp_lr_minus * self.traces[e_post] * fire_pre)
            self.edge_w += delta
            xp.clip(self.edge_w, self.w_min, self.w_max, out=self.edge_w)

            pulse_out[...] = xp.bincount(
                e_post, weights=self.edge_w * fire_pre, minlength=N)
        else:
            pulse_out.fill(0)

        # 6. 发放神经元迹 +1
        self.traces += fire_f

        # 8. 滑动窗口 + 动态阈值
        old_row = self.pot_history[self.hist_ptr]
        self.roll_sum += self.membrane - old_row
        self.pot_history[self.hist_ptr] = self.membrane
        filled = min(self.time_step + 1, self.window_size)
        xp.multiply(self.roll_sum, self.adaptation_rate / filled, out=self.thresholds)
        self.thresholds += self.base_threshold

        self.hist_ptr = (self.hist_ptr + 1) % self.window_size
        self.time_step += 1

    # -------------------------------------------------------------------
    # 设备管理
    # -------------------------------------------------------------------

    def _select_device(self, device: str, bench_steps: int):
        cp = _try_load_cupy()
        if cp is None:
            if device == "gpu":
                warnings.warn("CuPy 或 GPU 不可用，回退到 CPU(numpy)。")
            return

        if device == "gpu":
            self._to_device(cp)
            self.device_name = "gpu(cupy)"
            return

        if bench_steps <= 0 or self.edge_pre.size == 0:
            if self.edge_pre.size >= 200_000 or self.n_total >= 50_000:
                self._to_device(cp)
                self.device_name = "gpu(cupy)"
            return

        snap = self._snapshot_state()

        for _ in range(3):
            self.step()
        self._restore_state(snap)
        t0 = time.perf_counter()
        for _ in range(bench_steps):
            self.step()
        t_cpu = (time.perf_counter() - t0) / bench_steps

        self._restore_state(snap)
        self._to_device(cp)
        for _ in range(3):
            self.step()
        self._restore_state(snap)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_steps):
            self.step()
        cp.cuda.Stream.null.synchronize()
        t_gpu = (time.perf_counter() - t0) / bench_steps

        if t_cpu <= t_gpu:
            self._to_device(np)
            self.device_name = "cpu(numpy)"
        else:
            self.device_name = "gpu(cupy)"
        self._restore_state(snap)

    def _to_device(self, xp):
        old = self._xp
        self._xp = xp
        for name in self._STATE_ARRAYS:
            arr = getattr(self, name)
            if old is not np:
                arr = arr.get()
            setattr(self, name, xp.asarray(arr))
        self.pulse_buf = [xp.asarray(b.get() if old is not np else b)
                          for b in self.pulse_buf]

    def move_to_device(self, device: str):
        if device == "cpu":
            self._to_device(np)
            self.device_name = "cpu(numpy)"
        elif device == "gpu":
            cp = _try_load_cupy()
            if cp is None:
                warnings.warn("CuPy 或 GPU 不可用，保持当前设备。")
                return
            self._to_device(cp)
            self.device_name = "gpu(cupy)"
        else:
            raise ValueError(f"device 须为 'cpu'/'gpu', 收到 {device!r}")

    def _snapshot_state(self) -> dict:
        to_np = (lambda a: a.get()) if self._xp is not np else (lambda a: a.copy())
        snap = {name: to_np(getattr(self, name)) for name in self._STATE_ARRAYS}
        snap["pulse_buf"] = [to_np(b) for b in self.pulse_buf]
        snap["hist_ptr"]  = self.hist_ptr
        snap["time_step"] = self.time_step
        return snap

    def _restore_state(self, snap: dict):
        xp = self._xp
        for name in self._STATE_ARRAYS:
            setattr(self, name, xp.asarray(snap[name]))
        self.pulse_buf = [xp.asarray(b) for b in snap["pulse_buf"]]
        self.hist_ptr  = snap["hist_ptr"]
        self.time_step = snap["time_step"]

    # ===================================================================
    # 标准化接口（去掉 read_output / inject_error）
    # ===================================================================

    def inject_input(self, signal):
        """向第一面注入 [-1,1] 强度图，直接叠加到对应神经元膜电位。"""
        flat = np.asarray(signal, dtype=self.dtype).reshape(-1)
        n = min(flat.size, self.n_face)
        self.membrane[self.first_start:self.first_start + n] += \
            self._xp.asarray(flat[:n])

    def read_action_spikes(self) -> np.ndarray:
        """动作神经元本轮是否发放。"""
        seg = self.fired[self.action_start:self.action_end]
        if self._xp is not np:
            seg = seg.get()
        return seg.copy()

    def read_action_potentials(self) -> np.ndarray:
        """动作神经元当前膜电位。"""
        seg = self.membrane[self.action_start:self.action_end]
        if self._xp is not np:
            seg = seg.get()
        return seg.copy()

    # ===================================================================
    # 信息查询
    # ===================================================================

    def get_connection_count(self) -> int:
        return int(self.edge_pre.size)

    def get_firing_rate(self) -> float:
        return float(self.fired.mean())

    def get_stats(self) -> dict:
        return {
            "time_step":      self.time_step,
            "device":         self.device_name,
            "n_connections":  self.get_connection_count(),
            "firing_rate":    round(self.get_firing_rate(), 4),
            "mean_threshold": round(float(self.thresholds.mean()), 4),
            "mean_membrane":  round(float(self.membrane.mean()), 4),
            "mean_trace":     round(float(self.traces.mean()), 4),
        }

    def get_layer_stats(self) -> dict:
        spans = {"first":  (self.first_start, self.first_end),
                 "middle": (self.middle_start, self.middle_end),
                 "action": (self.action_start, self.action_end)}
        stats = {}
        for name, (a, b) in spans.items():
            n = b - a
            stats[name] = {
                "n_neurons":      n,
                "firing_rate":    round(float(self.fired[a:b].mean()), 4) if n else 0.0,
                "mean_membrane":  round(float(self.membrane[a:b].mean()), 4) if n else 0.0,
                "mean_threshold": round(float(self.thresholds[a:b].mean()), 4) if n else 0.0,
            }
        return stats


# ===========================================================================
# 第二部分：后台运行器
# ===========================================================================

class EngineRunner:
    """后台线程持续运行引擎，外部随时读写。"""

    def __init__(self, engine: PulseEngine, steps_per_sec: Optional[float] = 200):
        self.engine = engine
        self.target_interval = (1.0 / steps_per_sec) if steps_per_sec else None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=3.0)

    def _run_loop(self):
        next_t = time.perf_counter()
        while self.running:
            with self._lock:
                self.engine.step()
            if self.target_interval is not None:
                next_t += self.target_interval
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.perf_counter()

    def inject_input(self, signal):
        with self._lock:
            self.engine.inject_input(signal)

    def read_action_spikes(self) -> np.ndarray:
        with self._lock:
            return self.engine.read_action_spikes()

    def read_action_potentials(self) -> np.ndarray:
        with self._lock:
            return self.engine.read_action_potentials()

    def get_stats(self) -> dict:
        with self._lock:
            return self.engine.get_stats()

    def get_layer_stats(self) -> dict:
        with self._lock:
            return self.engine.get_layer_stats()


# ===========================================================================
# 第三部分：演示用占位坐标（非引擎组成部分）
# ===========================================================================

def demo_placeholder_galaxy_coords(n: int, seed: int = 42) -> np.ndarray:
    """
    ⚠ 占位数据源，仅用于 demo/测试。
    生成 n 个 [0,1]^3 的"星系状"坐标（大星团 + 小星群 + 均匀补齐）。
    真实部署时用实际数据替换；引擎本身不依赖此函数。
    """
    rng = np.random.default_rng(seed)
    coords: List[np.ndarray] = []

    n_clusters = max(1, n // 150)
    for _ in range(n_clusters):
        center  = rng.uniform(0.05, 0.95, 3)
        spread  = rng.uniform(0.02, 0.06)
        n_in    = int(rng.integers(40, 120))
        axis    = int(rng.integers(0, 3))
        stretch = float(rng.uniform(3.0, 8.0))
        for _ in range(n_in):
            off = rng.normal(0, spread, 3)
            off[axis] *= stretch
            coords.append(np.clip(center + off, 0, 1))
        if len(coords) >= n:
            break

    n_groups = max(1, n // 30)
    for _ in range(n_groups):
        center = rng.uniform(0, 1, 3)
        spread = rng.uniform(0.01, 0.025)
        n_in   = int(rng.integers(5, 25))
        for _ in range(n_in):
            coords.append(np.clip(center + rng.normal(0, spread, 3), 0, 1))
        if len(coords) >= n:
            break

    while len(coords) < n:
        coords.append(rng.uniform(0, 1, 3))

    return np.array(coords[:n])


# ===========================================================================
# 第四部分：演示
# ===========================================================================

def demo_basic():
    print("=" * 70)
    print("SUSNN 核心引擎 —— 纯动作输出演示")
    print("=" * 70)

    galaxy = demo_placeholder_galaxy_coords(n=500, seed=42)
    engine = PulseEngine(face_rows=16, face_cols=16, n_middle=500, n_action=4,
                         connection_radius=4.0, space_depth=12.0,
                         window_size=100, stdp_tau=15.0, seed=42,
                         galaxy_coords=galaxy)
    print(f"\n运行设备   : {engine.device_name}")
    print(f"总神经元数 : {engine.n_total}  "
          f"(第一面 {engine.n_face} + 中间 {engine.n_middle} "
          f"+ 动作 {engine.n_action})")
    print(f"初始连接数 : {engine.get_connection_count()}")

    def make_input(t: int) -> np.ndarray:
        rows, cols = engine.face_rows, engine.face_cols
        cx = cols / 2 + cols / 4 * np.sin(t * 0.05)
        cy = rows / 2 + rows / 4 * np.cos(t * 0.07)
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        sig = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.0 ** 2))
        return (sig * 2 - 1).astype(np.float32)

    print("\n--- 运行 2000 步 ---")
    n_steps = 2000
    t0 = time.perf_counter()
    for t in range(n_steps):
        engine.inject_input(make_input(t) * 0.5)
        engine.step()
        if (t + 1) % 500 == 0:
            stats = engine.get_stats()
            print(f"  step={stats['time_step']:5d} | "
                  f"fire={stats['firing_rate']:.3f} | "
                  f"thr={stats['mean_threshold']:.3f} | "
                  f"conns={stats['n_connections']}")
    dt = time.perf_counter() - t0
    print(f"\n  {n_steps} 步用时 {dt:.2f}s ({n_steps / dt:.0f} steps/s)")

    print("\n--- 各层状态 ---")
    for name, s in engine.get_layer_stats().items():
        print(f"  {name:6s}: N={s['n_neurons']:5d} | "
              f"fire={s['firing_rate']:.4f} | "
              f"mem={s['mean_membrane']:+.4f} | "
              f"thr={s['mean_threshold']:.4f}")

    spikes = engine.read_action_spikes()
    pots = engine.read_action_potentials()
    print("\n--- 动作神经元 ---")
    for i in range(len(spikes)):
        print(f"  动作{i}: 发放={bool(spikes[i])} | 膜电位={pots[i]:+.3f}")

    # 观察 STDP 剪枝效果
    w = engine.edge_w if engine._xp is np else engine.edge_w.get()
    print(f"\n--- 权重分布 ---")
    print(f"  总边数  : {w.size}")
    print(f"  ≈0 的边 : {(w <= 1e-6).sum()}  (被 LTD 压到 0)")
    print(f"  min/med/max: {w.min():.4f} / {np.median(w):.4f} / {w.max():.4f}")
    print("=" * 70)


def demo_background_runner():
    print("\n" + "=" * 70)
    print("后台运行器演示")
    print("=" * 70)

    galaxy = demo_placeholder_galaxy_coords(n=300, seed=123)
    engine = PulseEngine(face_rows=12, face_cols=12, n_middle=300,
                         n_action=4, seed=123, galaxy_coords=galaxy)
    runner = EngineRunner(engine, steps_per_sec=200)
    runner.start()
    print(f"引擎后台线程已启动 (200 steps/s, 设备 {engine.device_name})")

    for i in range(50):
        val = 0.5 * np.sin(i * 0.2)
        inp = np.full((engine.face_rows, engine.face_cols), val, dtype=np.float32)
        runner.inject_input(inp * 0.3)
        time.sleep(0.1)
        if (i + 1) % 10 == 0:
            stats = runner.get_stats()
            print(f"  Cycle {i+1:3d} | Step={stats['time_step']:6d} | "
                  f"FireRate={stats['firing_rate']:.3f} | "
                  f"Conns={stats['n_connections']}")

    runner.stop()
    print("引擎后台线程已停止")
    print("=" * 70)


if __name__ == "__main__":
    demo_basic()
    demo_background_runner()
