"""
核心脉冲引擎 —— 高性能向量化版
============================================================
在保持设计逻辑与对外接口不变的前提下优化:
1. 连接存储: Dict[int, Dict[int, float]] → 平行边数组
2. 一个时间步内所有神经元同步更新 —— 即设计文档所述
   "资源足够时直接一次性计算(同时扫描)"的语义。
   (原实现中脉冲写入 next 缓冲、下一步才作用, 步内本就互不依赖;
    唯一步内耦合只有"先扫描者的迹增量被后扫描者的 STDP 读到",
    而扫描顺序本身就是每步随机洗牌的。)
3. CPU: numpy 全向量化; GPU: CuPy 同构后端;
   device='auto' 时初始化阶段自动基准测试择优。
4. 睡眠态结构可塑性(生长/剪枝)同样完全向量化。

对外接口与原版一致:
    step() inject_input() read_output() inject_error()
    read_action_spikes() read_action_potentials()
    enter_sleep() exit_sleep()
    get_connection_count() get_firing_rate() get_stats() get_layer_stats()

内部结构变化(如有外部代码依赖请注意):
    conn_out / conn_in / pulse_active / pulse_next / scan_order / pot_history_sum
      → edge_pre / edge_post / edge_w / pulse_buf / roll_sum
    pot_history 形状由 (N, W) 改为 (W, N) (每步写一行, 内存连续)
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Optional, Dict, List, Tuple

import numpy as np


# ===========================================================================
# 第零部分: CuPy 惰性加载与通用工具
# ===========================================================================

_CUPY_CACHE: Dict[str, object] = {}


def _try_load_cupy():
    """尝试加载 CuPy 并确认存在可用 GPU; 失败返回 None(自动回退 CPU)。"""
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


def _keys_exist(keys: np.ndarray, q: np.ndarray) -> np.ndarray:
    """有序键数组 keys 中批量判断 q 的每个键是否存在(searchsorted)。"""
    if keys.size == 0 or q.size == 0:
        return np.zeros(q.size, dtype=bool)
    pos = np.searchsorted(keys, q)
    np.clip(pos, 0, keys.size - 1, out=pos)
    return keys[pos] == q


def _block_cross(starts: np.ndarray, counts: np.ndarray,
                 u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量单元格叉积: 第 k 组 = "单元格 u[k] 的全部成员 × 单元格 v[k] 的全部成员"。
    starts/counts: 每个单元格在(按格排序的)成员数组中的起始下标与数量。
    返回成员下标对, ib)。
    """
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
# 第一部分: 核心脉冲引擎(向量化)
# ===========================================================================

class PulseEngine:
    """
    核心脉冲引擎(向量化实现, 对外接口与原版一致)。
    """

    LAYER_FIRST  = 0
    LAYER_SECOND = 1
    LAYER_MIDDLE = 2
    LAYER_ACTION = 3

    # 需要在 CPU/GPU 之间迁移 / 快照恢复的全部状态数组
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
                 new_conn_weight: float = 0.1,
                 hebb_threshold: float = 0.05,
                 prune_threshold: float = 0.005,
                 galaxy_coords: Optional[np.ndarray] = None,
                 seed: int = 42,
                 device: str = "auto",          # 新增: 'auto' | 'cpu' | 'gpu'
                 dtype=np.float64,              # 新增: 状态数值精度
                 auto_bench_steps: int = 30):   # 新增: auto 模式基准步数
        self.rng = np.random.default_rng(seed)

        # ---- 参数(与原版一致) ----
        self.face_rows       = face_rows
        self.face_cols       = face_cols
        self.n_face          = face_rows * face_cols
        self.n_middle        = n_middle
        self.n_action        = n_action
        self.n_total         = self.n_face * 2 + n_middle + n_action
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
        self.new_conn_weight = new_conn_weight
        self.hebb_threshold  = hebb_threshold
        self.prune_threshold = prune_threshold

        # ---- 新增 ----
        self.dtype = np.dtype(dtype)
        self._xp = np                       # 数组模块: numpy 或 cupy
        self.device_name = "cpu(numpy)"

        # ---- 构建网络 ----
        self._build_network(galaxy_coords)
        self._init_connections()
        self._init_state()
        self._trace_decay = float(np.exp(-1.0 / self.stdp_tau))

        # ---- 设备选择 ----
        if device in ("auto", "gpu"):
            self._select_device(device, auto_bench_steps)
        elif device != "cpu":
            raise ValueError(f"device 须为 'auto'/'cpu'/'gpu', 收到 {device!r}")

    # -------------------------------------------------------------------
    # 网络构建: 空间坐标(与原版一致)
    # -------------------------------------------------------------------

    def _build_network(self, galaxy_coords: Optional[np.ndarray]):
        positions: List[List[float]] = []
        layer_ids: List[int] = []
        idx = 0

        for r in range(self.face_rows):
            for c in range(self.face_cols):
                positions.append([float(c), float(r), 0.0])
                layer_ids.append(self.LAYER_FIRST)
        self.first_start, self.first_end = 0, idx + self.n_face
        idx = self.first_end

        for r in range(self.face_rows):
            for c in range(self.face_cols):
                positions.append([float(c), float(r), self.space_depth])
                layer_ids.append(self.LAYER_SECOND)
        self.second_start, self.second_end = idx, idx + self.n_face
        idx = self.second_end

        if galaxy_coords is not None and len(galaxy_coords) >= self.n_middle:
            coords = self._scale_galaxy_coords(galaxy_coords[:self.n_middle])
        else:
            coords = self._generate_galaxy_coords(self.n_middle)

        for i in range(self.n_middle):
            positions.append(coords[i].tolist())
            layer_ids.append(self.LAYER_MIDDLE)
        self.middle_start, self.middle_end = idx, idx + self.n_middle
        idx = self.middle_end

        for i in range(self.n_action):
            positions.append([self.face_cols + 2.0 + i * 1.5,
                              self.face_rows / 2.0, self.space_depth])
            layer_ids.append(self.LAYER_ACTION)
        idx += self.n_action
        self.action_start, self.action_end = self.middle_end, idx

        assert idx == self.n_total
        self.positions = np.array(positions, dtype=np.float32)
        self.layer_ids = np.array(layer_ids, dtype=np.int32)

        self.is_input_neuron  = (self.layer_ids == self.LAYER_FIRST)
        self.is_pred_neuron   = (self.layer_ids == self.LAYER_SECOND)
        self.is_action_neuron = (self.layer_ids == self.LAYER_ACTION)
        self.is_middle_neuron = (self.layer_ids == self.LAYER_MIDDLE)

    def _scale_galaxy_coords(self, coords: np.ndarray) -> np.ndarray:
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

    def _generate_galaxy_coords(self, n: int) -> np.ndarray:
        coords: List[np.ndarray] = []

        n_clusters = max(1, n // 150)
        for _ in range(n_clusters):
            center  = self.rng.uniform(0.05, 0.95, 3)
            spread  = self.rng.uniform(0.02, 0.06)
            n_in    = int(self.rng.integers(40, 120))
            axis    = int(self.rng.integers(0, 3))
            stretch = float(self.rng.uniform(3.0, 8.0))
            for _ in range(n_in):
                off = self.rng.normal(0, spread, 3)
                off[axis] *= stretch
                coords.append(np.clip(center + off, 0, 1))
            if len(coords) >= n:
                break

        n_groups = max(1, n // 30)
        for _ in range(n_groups):
            center = self.rng.uniform(0, 1, 3)
            spread = self.rng.uniform(0.01, 0.025)
            n_in   = int(self.rng.integers(5, 25))
            for _ in range(n_in):
                coords.append(np.clip(center + self.rng.normal(0, spread, 3), 0, 1))
            if len(coords) >= n:
                break

        while len(coords) < n:
            coords.append(self.rng.uniform(0, 1, 3))

        return self._scale_galaxy_coords(np.array(coords[:n]))

    # -------------------------------------------------------------------
    # 空间近邻配对(向量化, 连接初始化与睡眠态生长共用)
    # -------------------------------------------------------------------

    def _pairs_within_radius(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        在给定(升序)神经元编号集合中, 返回所有空间距离 < connection_radius
        的无序对, hi), lo < hi (int64)。
        算法: 均匀网格(格边长=连接半径) + 27 邻域配对, 全程向量化。
        """
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

        # --- 跨单元格: 13 个规范偏移(o 与 -o 只取一个) ---
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

        # --- 同一单元格内部: 全组合后保留 lo < hi ---
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
    # 连接初始化(与原实现同分布: 距离<r 的无序对, 15% 建边, 小索引→大索引)
    # -------------------------------------------------------------------

    def _init_connections(self):
        ids = np.arange(self.n_total, dtype=np.int64)
        lo, hi = self._pairs_within_radius(ids)
        if lo.size:
            keep = self.rng.random(lo.size) < 0.15
            lo, hi = lo[keep], hi[keep]
        w = (self.rng.uniform(0.02, 0.3, size=lo.size).astype(self.dtype, copy=False)
             if lo.size else np.zeros(0, dtype=self.dtype))
        self.edge_pre  = lo.astype(np.int32)   # 每条边: 突触前
        self.edge_post = hi.astype(np.int32)   # 每条边: 突触后
        self.edge_w    = w                     # 每条边: 权重
        self._rebuild_edge_keys(self.edge_pre, self.edge_post)

    def _rebuild_edge_keys(self, e_pre=None, e_post=None):
        """维护有序有向边键(pre*N+post), 供睡眠态生长的批量存在性检查。"""
        if e_pre is None:
            e_pre  = self.edge_pre  if self._xp is np else self.edge_pre.get()
            e_post = self.edge_post if self._xp is np else self.edge_post.get()
        if e_pre.size:
            self._edge_keys = np.sort(
                e_pre.astype(np.int64) * self.n_total + e_post.astype(np.int64))
        else:
            self._edge_keys = np.empty(0, dtype=np.int64)

    # -------------------------------------------------------------------
    # 状态初始化
    # -------------------------------------------------------------------

    def _init_state(self):
        N, dt = self.n_total, self.dtype
        self.membrane   = np.zeros(N, dtype=dt)
        self.thresholds = np.full(N, self.base_threshold, dtype=dt)
        self.traces     = np.zeros(N, dtype=dt)
        # (W, N) 布局: 每个时间步写一整行(内存连续); 历史值 float32(与原版一致)
        self.pot_history = np.zeros((self.window_size, N), dtype=np.float32)
        self.roll_sum    = np.zeros(N, dtype=dt)      # 滑动窗口滚动和
        self.hist_ptr    = 0
        self.pulse_buf   = [np.zeros(N, dtype=dt), np.zeros(N, dtype=dt)]  # 双缓冲
        self._buf        = 0
        self.fired       = np.zeros(N, dtype=bool)
        self.time_step   = 0
        self.is_sleeping = False

    # -------------------------------------------------------------------
    # 核心: 一个时间步(所有神经元同步更新一次)
    # -------------------------------------------------------------------

    def step(self):
        """
        执行一个时间步 = 所有神经元"同时"各更新一次(设计文档的同时扫描语义)。
        流程与原逐神经元扫描逐条对应:
        1. 交换脉冲双缓冲(上一步发出的脉冲本步到达)
        2. STDP 迹全局衰减
        3. 到达脉冲叠加到膜电位(外部注入已由 inject_* 叠加)
        4. 阈值判定 → 发放/仅积分, 发放者减法重置(保留超出部分)
        5. STDP: 突触后发放 → LTP(按突触前迹); 突触前发放 → LTD(按突触后迹)
        6. 发放神经元迹 +1(置于 STDP 之后: 同步语义下本步发放不计入本步 STDP)
        7. 发放脉冲按(STDP 更新后的)权重散射进下一时间步缓冲
        8. 滑动窗口滚动和 + 动态阈值
        9. 睡眠态: 结构可塑性(生长与剪枝)
        """
        xp = self._xp
        N = self.n_total

        # 1. 双缓冲交换: 本步读 cur, 写 nxt
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

        if self.edge_pre.size:
            e_pre, e_post = self.edge_pre, self.edge_post

            # 5. STDP(迹 = 本步衰减后、发放增量前的值)
            fire_pre  = fire_f[e_pre]      # 每条边的突触前是否发放
            fire_post = fire_f[e_post]     # 每条边的突触后是否发放
            delta = (self.stdp_lr_plus  * self.traces[e_pre]  * fire_post
                     - self.stdp_lr_minus * self.traces[e_post] * fire_pre)
            self.edge_w += delta
            xp.clip(self.edge_w, self.w_min, self.w_max, out=self.edge_w)

            # 7. 脉冲散射(bincount 按目标聚合; 未发放突触前贡献为 0)
            pulse_out[...] = xp.bincount(
                e_post, weights=self.edge_w * fire_pre, minlength=N)
        else:
            pulse_out.fill(0)

        # 6. 发放迹 +1
        #   (如需"同步发放也计入对方 STDP"的变体, 把本行移到第 5 步之前即可)
        self.traces += fire_f

        # 8. 滑动窗口(滚动和) + 动态阈值
        old_row = self.pot_history[self.hist_ptr]
        self.roll_sum += self.membrane - old_row
        self.pot_history[self.hist_ptr] = self.membrane
        filled = min(self.time_step + 1, self.window_size)
        xp.multiply(self.roll_sum, self.adaptation_rate / filled, out=self.thresholds)
        self.thresholds += self.base_threshold

        # 9. 睡眠态结构可塑性
        if self.is_sleeping:
            self._structural_plasticity()

        self.hist_ptr = (self.hist_ptr + 1) % self.window_size
        self.time_step += 1

    # -------------------------------------------------------------------
    # 睡眠态结构可塑性(向量化; GPU 模式下自动在 CPU 上做再同步回设备)
    # -------------------------------------------------------------------

    def _structural_plasticity(self):
        """
        与原实现一致:
        - 生长: Hebb 痕迹超阈值的近邻对, 双向均无连接时以 10% 概率
          随机方向建立权重为 new_conn_weight 的新连接
        - 剪枝: 删除权重 < prune_threshold 的连接
        """
        xp = self._xp
        if xp is np:
            e_pre, e_post, e_w = self.edge_pre, self.edge_post, self.edge_w
            traces = self.traces
        else:
            e_pre, e_post, e_w = self.edge_pre.get(), self.edge_post.get(), self.edge_w.get()
            traces = self.traces.get()

        N = self.n_total
        added = pruned = 0

        # ---- 生长 ----
        active = np.flatnonzero(traces > self.hebb_threshold)
        if active.size > 1:
            lo, hi = self._pairs_within_radius(active)
            if lo.size:
                keys = self._edge_keys
                exists = (_keys_exist(keys, lo * N + hi)
                          | _keys_exist(keys, hi * N + lo))
                lo, hi = lo[~exists], hi[~exists]
            if lo.size:
                grow = self.rng.random(lo.size) < 0.1
                lo, hi = lo[grow], hi[grow]
            if lo.size:
                fwd = self.rng.random(lo.size) < 0.5          # 随机方向
                new_pre  = np.where(fwd, lo, hi).astype(np.int32)
                new_post = np.where(fwd, hi, lo).astype(np.int32)
                e_pre  = np.concatenate([e_pre, new_pre])
                e_post = np.concatenate([e_post, new_post])
                e_w    = np.concatenate(
                    [e_w, np.full(new_pre.size, self.new_conn_weight, dtype=e_w.dtype)])
                added = int(new_pre.size)

        # ---- 剪枝 ----
        dead = e_w < self.prune_threshold
        if dead.any():
            alive = ~dead
            e_pre, e_post, e_w = e_pre[alive], e_post[alive], e_w[alive]
            pruned = int(dead.sum())

        if added or pruned:
            self.edge_pre  = xp.asarray(e_pre)
            self.edge_post = xp.asarray(e_post)
            self.edge_w    = xp.asarray(e_w)
            self._rebuild_edge_keys(e_pre, e_post)

        return added, pruned

    # -------------------------------------------------------------------
    # 设备管理(CPU/GPU)
    # -------------------------------------------------------------------

    def _select_device(self, device: str, bench_steps: int):
        cp = _try_load_cupy()
        if cp is None:
            if device == "gpu":
                warnings.warn("CuPy 或 GPU 不可用, 回退到 CPU(numpy)。")
            return

        if device == "gpu":
            self._to_device(cp)
            self.device_name = "gpu(cupy)"
            return

        # ---- device == "auto" ----
        if bench_steps <= 0 or self.edge_pre.size == 0:
            # 无基准时的规模启发: 足够大的网络才值得上 GPU
            if self.edge_pre.size >= 200_000 or self.n_total >= 50_000:
                self._to_device(cp)
                self.device_name = "gpu(cupy)"
            return

        snap = self._snapshot_state()

        # CPU 基准
        for _ in range(3):
            self.step()                                  # 预热
        self._restore_state(snap)
        t0 = time.perf_counter()
        for _ in range(bench_steps):
            self.step()
        t_cpu = (time.perf_counter() - t0) / bench_steps

        # GPU 基准
        self._restore_state(snap)
        self._to_device(cp)
        for _ in range(3):
            self.step()                                  # 预热(编译内核)
        self._restore_state(snap)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        for _ in range(bench_steps):
            self.step()
        cp.cuda.Stream.null.synchronize()
        t_gpu = (time.perf_counter() - t0) / bench_steps

        # 择优, 并恢复到基准前的初始状态
        if t_cpu <= t_gpu:
            self._to_device(np)
            self.device_name = "cpu(numpy)"
        else:
            self.device_name = "gpu(cupy)"
        self._restore_state(snap)

    def _to_device(self, xp):
        """把全部状态数组迁移到指定数组模块(np 或 cupy)。"""
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
        """运行时手动迁移('cpu'/'gpu'); 请在后台线程未运行时调用。"""
        if device == "cpu":
            self._to_device(np)
            self.device_name = "cpu(numpy)"
        elif device == "gpu":
            cp = _try_load_cupy()
            if cp is None:
                warnings.warn("CuPy 或 GPU 不可用, 保持当前设备。")
                return
            self._to_device(cp)
            self.device_name = "gpu(cupy)"
        else:
            raise ValueError(f"device 须为 'cpu'/'gpu', 收到 {device!r}")

    def _snapshot_state(self) -> dict:
        to_np = (lambda a: a.get()) if self._xp is not np else (lambda a: a.copy())
        snap = {name: to_np(getattr(self, name)) for name in self._STATE_ARRAYS}
        snap["pulse_buf"] = [to_np(b) for b in self.pulse_buf]
        snap["hist_ptr"] = self.hist_ptr
        snap["time_step"] = self.time_step
        return snap

    def _restore_state(self, snap: dict):
        xp = self._xp
        for name in self._STATE_ARRAYS:
            setattr(self, name, xp.asarray(snap[name]))
        self.pulse_buf = [xp.asarray(b) for b in snap["pulse_buf"]]
        self.hist_ptr = snap["hist_ptr"]
        self.time_step = snap["time_step"]

    # ===================================================================
    # 标准化接口(与原版一致)
    # ===================================================================

    def inject_input(self, signal):
        """向第一面注入 [-1,1] 强度图, 直接叠加到对应神经元膜电位。"""
        flat = np.asarray(signal, dtype=self.dtype).reshape(-1)
        n = min(flat.size, self.n_face)
        self.membrane[self.first_start:self.first_start + n] += self._xp.asarray(flat[:n])

    def read_output(self) -> np.ndarray:
        """读取第二面当前膜电位(连续标量), 返回。"""
        seg = self.membrane[self.second_start:self.second_end]
        if self._xp is not np:
            seg = seg.get()
        return seg.reshape(self.face_rows, self.face_cols).copy()

    def inject_error(self, error):
        """向第二面(除动作神经元)注入误差信号, 直接叠加到膜电位。"""
        flat = np.asarray(error, dtype=self.dtype).reshape(-1)
        n = min(flat.size, self.n_face)
        self.membrane[self.second_start:self.second_start + n] += self._xp.asarray(flat[:n])

    def read_action_spikes(self) -> np.ndarray:
        """动作神经元本轮是否发放(布尔数组)。"""
        seg = self.fired[self.action_start:self.action_end]
        if self._xp is not np:
            seg = seg.get()
        return seg.copy()

    def read_action_potentials(self) -> np.ndarray:
        """动作神经元当前膜电位(连续值)。"""
        seg = self.membrane[self.action_start:self.action_end]
        if self._xp is not np:
            seg = seg.get()
        return seg.copy()

    # ===================================================================
    # 睡眠控制 / 信息查询
    # ===================================================================

    def enter_sleep(self):
        self.is_sleeping = True

    def exit_sleep(self):
        self.is_sleeping = False

    def get_connection_count(self) -> int:
        return int(self.edge_pre.size)

    def get_firing_rate(self) -> float:
        return float(self.fired.mean())

    def get_stats(self) -> dict:
        return {
            "time_step":      self.time_step,
            "sleeping":       self.is_sleeping,
            "device":         self.device_name,
            "n_connections":  self.get_connection_count(),
            "firing_rate":    round(self.get_firing_rate(), 4),
            "mean_threshold": round(float(self.thresholds.mean()), 4),
            "mean_membrane":  round(float(self.membrane.mean()), 4),
            "mean_trace":     round(float(self.traces.mean()), 4),
        }

    def get_layer_stats(self) -> dict:
        spans = {"first":  (self.first_start, self.first_end),
                 "second": (self.second_start, self.second_end),
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
# 第二部分: 后台运行器
# ===========================================================================

class EngineRunner:
    """
    后台线程持续运行引擎, 外部随时读写。接口与原版一致;
    计时改为绝对时刻调度, 避免长期漂移。steps_per_sec 传 None 时自由运行。
    """

    def __init__(self, engine: PulseEngine, steps_per_sec: float = 200):
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
                    next_t = time.perf_counter()   # 落后过多时重新对时

    # ---- 线程安全接口(与原版一致) ----
    def inject_input(self, signal):
        with self._lock:
            self.engine.inject_input(signal)

    def read_output(self) -> np.ndarray:
        with self._lock:
            return self.engine.read_output()

    def inject_error(self, error):
        with self._lock:
            self.engine.inject_error(error)

    def read_action_spikes(self) -> np.ndarray:
        with self._lock:
            return self.engine.read_action_spikes()

    def read_action_potentials(self) -> np.ndarray:
        with self._lock:
            return self.engine.read_action_potentials()

    def enter_sleep(self):
        with self._lock:
            self.engine.enter_sleep()

    def exit_sleep(self):
        with self._lock:
            self.engine.exit_sleep()

    def get_stats(self) -> dict:
        with self._lock:
            return self.engine.get_stats()

    def get_layer_stats(self) -> dict:
        with self._lock:
            return self.engine.get_layer_stats()


# ===========================================================================
# 第三部分: 演示(与原 demo 逻辑相同, 附计时)
# ===========================================================================

def demo_prediction_learning():
    print("=" * 70)
    print("核心脉冲引擎(向量化版) —— 预测误差学习演示")
    print("=" * 70)

    engine = PulseEngine(face_rows=16, face_cols=16, n_middle=500, n_action=4,
                         connection_radius=4.0, space_depth=12.0,
                         window_size=100, stdp_tau=15.0, seed=42)
    print(f"\n运行设备   : {engine.device_name}")
    print(f"总神经元数 : {engine.n_total}")
    print(f"初始连接数 : {engine.get_connection_count()}")

    def make_input(t: int) -> np.ndarray:
        rows, cols = engine.face_rows, engine.face_cols
        cx = cols / 2 + cols / 4 * np.sin(t * 0.05)
        cy = rows / 2 + rows / 4 * np.cos(t * 0.07)
        y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        sig = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.0 ** 2))
        return (sig * 2 - 1).astype(np.float32)

    print("\n--- 清醒态训练 ---")
    n_wake, err_hist = 2000, []
    t0 = time.perf_counter()
    for t in range(n_wake):
        prediction = engine.read_output()
        current_input = make_input(t)
        error = np.clip(prediction - current_input, -1, 1)
        engine.inject_error(error * 0.5)
        engine.inject_input(current_input * 0.5)
        engine.step()
        err_hist.append(float(np.mean(error ** 2)))
    dt = time.perf_counter() - t0
    print(f"  {n_wake} 步用时 {dt:.2f}s ({n_wake / dt:.0f} steps/s)")
    print(f"  训练结束 | 最终 MSE = {np.mean(err_hist[-100:]):.4f}")

    print("\n--- 睡眠态(结构可塑性) ---")
    conn_before = engine.get_connection_count()
    engine.enter_sleep()
    t0 = time.perf_counter()
    for _ in range(500):
        engine.step()
    dt = time.perf_counter() - t0
    engine.exit_sleep()
    conn_after = engine.get_connection_count()
    print(f"  500 步用时 {dt:.2f}s | 连接数 {conn_before} → {conn_after} "
          f"({conn_after - conn_before:+d})")

    print("\n--- 睡眠后继续训练 ---")
    err_hist2 = []
    for t in range(1000):
        prediction = engine.read_output()
        current_input = make_input(n_wake + t)
        error = np.clip(prediction - current_input, -1, 1)
        engine.inject_error(error * 0.5)
        engine.inject_input(current_input * 0.5)
        engine.step()
        err_hist2.append(float(np.mean(error ** 2)))
    print(f"  睡眠前 MSE : {np.mean(err_hist[-100:]):.4f}")
    print(f"  睡眠后 MSE : {np.mean(err_hist2[-100:]):.4f}")

    spikes = engine.read_action_spikes()
    pots = engine.read_action_potentials()
    print("\n--- 动作神经元 ---")
    for i in range(len(spikes)):
        print(f"  动作{i}: 发放={bool(spikes[i])} | 膜电位={pots[i]:.3f}")
    print("=" * 70)


def demo_background_runner():
    print("\n" + "=" * 70)
    print("后台运行器演示")
    print("=" * 70)
    engine = PulseEngine(face_rows=12, face_cols=12, n_middle=300,
                         n_action=4, seed=123)
    runner = EngineRunner(engine, steps_per_sec=200)
    runner.start()
    print(f"引擎后台线程已启动 (200 steps/s, 设备 {engine.device_name})")

    for i in range(50):
        pred = runner.read_output()
        val = 0.5 * np.sin(i * 0.2)
        inp = np.full((engine.face_rows, engine.face_cols), val, dtype=np.float32)
        error = np.clip(pred - inp, -1, 1)
        runner.inject_error(error * 0.3)
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
    demo_prediction_learning()
    demo_background_runner()