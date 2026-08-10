"""
波-神经元自统一神经网络 - 核心脉冲引擎

按设计稿实现：
1. 三维空间结构：Face1(z=0,输入) + Hidden(z=0.2~0.8,星图分布) + Face2(z=1,输出)
2. IF神经元：积分-放电-减法重置（保留惯性，天然不应期）
3. 动态阈值：膜电位滑动窗口自适应（活跃→升高，沉寂→降低）
4. 永恒扫描器：按 (z, xy角度) 排序，Face1→Hidden→Face2 螺旋扫描
5. STDP：基于扫描时序差值的脉冲时间依赖可塑性
6. 清醒/睡眠态：睡眠期增加结构生长与剪枝
7. 标准化整面接口
8. 动作执行扩展：可读取任意神经元膜电位

依赖：numpy, scipy
"""

import numpy as np
import random
from scipy.spatial.distance import cdist


class PulseEngine:
    """核心脉冲引擎"""

    def __init__(self, num_face=769, num_hidden=100,
                 conn_radius=0.25, n_init_conn=2000, seed=42):
        """
        参数:
            num_face:    每面神经元数（Face1=Face2=num_face）
            num_hidden:  中间隐藏层神经元数
            conn_radius: 连接半径（3D空间中归一化距离）
            n_init_conn: 初始连接数
            seed:        随机种子
        """
        random.seed(seed)
        np.random.seed(seed)

        self.nf = num_face
        self.nh = num_hidden
        self.N = num_face * 2 + num_hidden
        self.conn_radius = conn_radius

        self._build_space()
        self._build_scan_order()
        self._init_connections(conn_radius, n_init_conn)

        # 状态
        self.V = np.zeros(self.N)              # 膜电位
        self.T = np.ones(self.N) * 1.0        # 放电阈值
        self.spike_cache = np.zeros(self.N)    # 脉冲缓存（单步延迟）
        self.last_spike = np.full(self.N, -1, dtype=int)

        # 动态阈值滑动窗口
        self.wsize = 50
        self.pbuf = np.zeros((self.N, self.wsize))
        self.wptr = 0
        self.wcount = 0  # 已填充的样本数

        # 睡眠态
        self.sleeping = False
        self.cofire = {}

        # STDP 参数
        self.ltp_rate = 0.01       # 同步LTP速率
        self.ltd_rate = 0.005      # 同步LTD速率
        self.ltp_rate_prev = 0.005  # 跨步LTP速率

        # 动态阈值参数
        self.th_base = 1.0
        self.th_alpha = 0.3
        self.th_min = 0.3
        self.th_max = 5.0

        self.step_num = 0

    # ==================== 空间结构 ====================
    def _build_space(self):
        """构建三维空间：Face1(z=0) + Hidden(星图) + Face2(z=1)"""
        n = self.nf

        # Face1: z=0 平面
        f1 = np.column_stack([
            np.random.uniform(0, 1, n),
            np.random.uniform(0, 1, n),
            np.zeros(n)
        ])
        # Face2: z=1 平面
        f2 = np.column_stack([
            np.random.uniform(0, 1, n),
            np.random.uniform(0, 1, n),
            np.ones(n)
        ])
        # Hidden: 宇宙大尺度结构（团簇+散落）
        nc = max(3, self.nh // 15)
        centers = np.random.uniform(0.1, 0.9, (nc, 3))
        centers[:, 2] = np.random.uniform(0.2, 0.8, nc)
        hid = np.zeros((self.nh, 3))
        for i in range(self.nh):
            c = centers[i % nc]
            hid[i] = np.clip(c + np.random.normal(0, 0.05, 3), 0, 1)

        self.pos = np.vstack([f1, f2, hid])
        self.s1 = slice(0, n)
        self.s2 = slice(n, 2 * n)
        self.sh = slice(2 * n, self.N)

    def _build_scan_order(self):
        """扫描顺序：按 (z×1000 + xy角度) 排序 → 螺旋扫描"""
        ang = np.arctan2(self.pos[:, 1], self.pos[:, 0])
        keys = self.pos[:, 2] * 1000 + ang
        self.scan_order = np.argsort(keys)
        self.scan_pos = np.empty(self.N, dtype=int)
        self.scan_pos[self.scan_order] = np.arange(self.N)

    def _init_connections(self, R, n_init):
        """初始化连接：连接半径内的随机子集"""
        D = cdist(self.pos, self.pos)
        mask = (D < R) & (D > 0.001)
        pairs = np.argwhere(mask)

        n_init = min(len(pairs), n_init)
        sel = np.random.choice(len(pairs), n_init, replace=False)
        pairs = pairs[sel]

        self.pre = pairs[:, 0].copy()
        self.post = pairs[:, 1].copy()
        self.w = np.random.uniform(0.01, 0.1, n_init).astype(np.float64)
        self.nc = n_init

        # 邻接表
        self.out_adj = [[] for _ in range(self.N)]
        self.in_adj = [[] for _ in range(self.N)]
        for i in range(n_init):
            self.out_adj[self.pre[i]].append((self.post[i], i))
            self.in_adj[self.post[i]].append((self.pre[i], i))

    # ==================== 标准化整面接口 ====================
    def input_face_1(self, signal):
        """向第一面传入信号（-1~1 强度图，叠加到膜电位）"""
        self.V[self.s1] += signal

    def input_face_2(self, error):
        """向第二面注入误差偏置（-1~1，叠加到膜电位）"""
        self.V[self.s2] += error

    def get_output_face_2(self):
        """从第二面读取输出（当前实时膜电位）"""
        return self.V[self.s2].copy()

    def get_potential(self, idx):
        """读取任意神经元膜电位（动作执行扩展）"""
        return self.V[idx]

    def get_state(self):
        """获取全局状态摘要"""
        active = np.sum(self.V > 0.01)
        return {
            'step': self.step_num,
            'connections': self.nc,
            'mean_V': float(self.V.mean()),
            'max_V': float(self.V.max()),
            'mean_T': float(self.T.mean()),
            'active_neurons': int(active),
            'sleeping': self.sleeping,
        }

    # ==================== 核心步进 ====================
    def step(self):
        """一个时间步 = 扫描指针完成一次对所有神经元的完整遍历"""
        # 1. 叠加缓存的脉冲电荷（单步延迟交付）
        self.V += self.spike_cache
        self.spike_cache[:] = 0

        # 2. 找到达阈值的神经元
        spiking = np.where(self.V >= self.T)[0]

        # 3. 逐个处理放电神经元
        for idx in spiking:
            idx = int(idx)

            # 3a. 脉冲分发：按权重放入目标缓存
            for (tgt, ci) in self.out_adj[idx]:
                self.spike_cache[tgt] += self.w[ci]

            # 3b. STDP：调整入向连接权重
            for (src, ci) in self.in_adj[idx]:
                ls = self.last_spike[src]
                if ls == self.step_num:
                    # 同步放电：扫描位置决定LTP/LTD
                    if self.scan_pos[src] < self.scan_pos[idx]:
                        self.w[ci] = min(1.0, self.w[ci] + self.ltp_rate)
                    else:
                        self.w[ci] = max(0.0, self.w[ci] - self.ltd_rate)
                elif ls == self.step_num - 1:
                    self.w[ci] = min(1.0, self.w[ci] + self.ltp_rate_prev)

            # 3c. 减法重置（保留惯性）
            self.V[idx] -= self.T[idx]
            self.last_spike[idx] = self.step_num

        # 4. 动态阈值
        self.pbuf[:, self.wptr] = self.V
        self.wptr = (self.wptr + 1) % self.wsize
        self.wcount = min(self.wcount + 1, self.wsize)
        if self.wcount >= 5:  # 至少5个样本后开始更新
            n = self.wcount
            avg = self.pbuf[:, :n].mean(axis=1)
            self.T = np.clip(self.th_base + self.th_alpha * avg,
                             self.th_min, self.th_max)

        # 5. 睡眠态结构可塑性
        if self.sleeping:
            self._sleep_plasticity(spiking)

        self.step_num += 1
        return spiking

    # ==================== 睡眠态结构可塑性 ====================
    def _sleep_plasticity(self, spiking):
        """睡眠态：共放电统计 + 结构生长 + 剪枝"""
        # 统计共同放电
        for i in range(len(spiking)):
            for j in range(i + 1, len(spiking)):
                a, b = int(spiking[i]), int(spiking[j])
                if abs(self.scan_pos[a] - self.scan_pos[b]) < 300:
                    key = (min(a, b), max(a, b))
                    self.cofire[key] = self.cofire.get(key, 0) + 1

        # 每20步：生长+剪枝
        if self.step_num % 20 == 0 and self.step_num > 0:
            # 生长
            for (a, b), count in list(self.cofire.items()):
                if count > 2 and len(self.out_adj[a]) < 80:
                    exists = any(t == b for t, _ in self.out_adj[a])
                    if not exists:
                        self._add_conn(a, b, 0.05)

            # 剪枝
            for ci in np.where(self.w < 0.001)[0]:
                self._del_conn(int(ci))

            self.cofire.clear()

    def _add_conn(self, p, q, wt):
        ci = len(self.w)
        self.pre = np.append(self.pre, p)
        self.post = np.append(self.post, q)
        self.w = np.append(self.w, wt)
        self.out_adj[p].append((q, ci))
        self.in_adj[q].append((p, ci))
        self.nc += 1

    def _del_conn(self, ci):
        if ci >= len(self.w) or self.w[ci] == 0:
            return
        p = int(self.pre[ci])
        q = int(self.post[ci])
        self.w[ci] = 0
        self.out_adj[p] = [(t, i) for t, i in self.out_adj[p] if i != ci]
        self.in_adj[q] = [(s, i) for s, i in self.in_adj[q] if i != ci]

    def enter_sleep(self):
        """进入睡眠态"""
        self.sleeping = True

    def exit_sleep(self):
        """退出睡眠态"""
        self.sleeping = False
