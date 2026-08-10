"""
波-神经元自统一神经网络 - 核心脉冲引擎 v2

按更新版设计稿实现：

一、核心脉冲引擎
1. 三维空间：Face1(z=0,输入) + Hidden(z=0.2~0.8,星图分布) + Face2(z=1,输出)
2. IF神经元：积分→放电→减法重置（保留惯性，天然不应期）
3. 动态阈值：膜电位滑动窗口平均自适应（活跃→升高，沉寂→降低）
4. 永恒扫描器：按三维空间顺序巡回，清醒/睡眠持续无差别运转
5. 脉冲单步延迟：发放后暂存缓存池，下一时间步交付目标
6. STDP：基于放电时间差，pre先于post→LTP，post先于pre→LTD
7. 清醒态 = 基础扫描 + STDP + 动态阈值（结构冻结）
8. 睡眠态 = 清醒态 + 结构生长（Hebb共放电痕迹）+ 剪枝（权重近零）

二、标准化整面接口
- input_face_1(signal)：-1~1强度图叠加到Face1膜电位
- get_output_face_2()：读取Face2当前膜电位作为预测输出
- input_face_2(error)：-1~1误差图注入Face2（跳过动作神经元）

三、动作执行扩展
- 动作神经元位于Face2，但不接收误差偏置
- 外部系统读取动作神经元的放电（脉冲），映射为控制指令
- 动作改变外部物理环境 → 影响下一帧Face1输入 → 间接闭环
"""

import numpy as np
import random
from scipy.spatial.distance import cdist


class PulseEngine:
    """核心脉冲引擎"""

    def __init__(self, num_face=770, num_hidden=100,
                 conn_radius=0.35, n_init_conn=3000, seed=42):
        """
        参数:
            num_face:    每面神经元数（Face1=Face2=num_face）
                         其中Face2可包含动作神经元
            num_hidden:  中间隐藏层神经元数
            conn_radius: 连接半径（3D归一化距离）
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

        # 神经元状态
        self.V = np.zeros(self.N)              # 膜电位
        self.T = np.ones(self.N) * 1.0         # 放电阈值
        self.spike_cache = np.zeros(self.N)    # 脉冲缓存（单步延迟）
        self.last_spike = np.full(self.N, -1, dtype=int)  # 最后放电步

        # 动态阈值滑动窗口
        self.wsize = 50
        self.pbuf = np.zeros((self.N, self.wsize))
        self.wptr = 0
        self.wcount = 0

        # 动态阈值参数
        self.th_base = 1.0
        self.th_alpha = 0.3
        self.th_min = 0.3
        self.th_max = 5.0

        # STDP参数（基于放电时间差）
        self.stdp_window = 10   # 时间窗口（步）
        self.stdp_tau = 5.0     # 指数衰减常数
        self.ltp_rate = 0.02    # LTP速率
        self.ltd_rate = 0.01    # LTD速率

        # 动作神经元
        self.action_neurons = set()
        self.error_mask = np.ones(num_face, dtype=bool)  # Face2误差掩码

        # 睡眠态
        self.sleeping = False
        self.cofire = {}
        self._free_list = []  # 可复用的死条目索引（防止数组膨胀）

        # 上一步放电记录
        self.last_spikes = np.array([], dtype=int)

        self.step_num = 0

    # ==================== 空间结构 ====================
    def _build_space(self):
        """构建三维空间：Face1(z=0) + Hidden(星图) + Face2(z=1)"""
        n = self.nf
        # Face1: z=0
        f1 = np.column_stack([
            np.random.uniform(0, 1, n),
            np.random.uniform(0, 1, n),
            np.zeros(n)
        ])
        # Face2: z=1
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
        self.s1 = slice(0, n)           # Face1
        self.s2 = slice(n, 2 * n)       # Face2
        self.sh = slice(2 * n, self.N)  # Hidden

    def _build_scan_order(self):
        """扫描顺序：按 (z, xy角度) 排序 → 螺旋扫描"""
        ang = np.arctan2(self.pos[:, 1], self.pos[:, 0])
        keys = self.pos[:, 2] * 1000 + ang
        self.scan_order = np.argsort(keys)

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
        """向第一面传入信号（-1~1 强度图，直接叠加到膜电位）"""
        self.V[self.s1] += signal

    def input_face_2(self, error):
        """向第二面注入误差偏置（-1~1，跳过动作神经元）"""
        self.V[self.s2] += error * self.error_mask

    def get_output_face_2(self):
        """从第二面读取输出（当前实时膜电位，连续标量）"""
        return self.V[self.s2].copy()

    # ==================== 动作执行扩展 ====================
    def set_action_neurons(self, face2_indices):
        """
        在Face2上标记动作神经元。
        动作神经元不接收误差偏置，外部系统读取其放电作为控制信号。

        参数:
            face2_indices: Face2上的神经元局部索引（0~nf-1）
        """
        self.action_neurons = set(face2_indices)
        self.error_mask = np.ones(self.nf, dtype=bool)
        for idx in face2_indices:
            self.error_mask[idx] = False

    def get_action_spikes(self):
        """
        获取上一步中动作神经元的放电情况。
        返回dict: {action_local_idx: spiked(bool)}
        """
        spike_set = set(self.last_spikes.tolist())
        result = {}
        for local_idx in self.action_neurons:
            global_idx = self.nf + local_idx  # Face2起始 + 局部索引
            result[local_idx] = global_idx in spike_set
        return result

    def get_spikes(self):
        """获取上一步放电的神经元全局索引列表"""
        return self.last_spikes

    def get_state(self):
        """获取全局状态摘要"""
        return {
            'step': self.step_num,
            'connections': self.nc,
            'mean_V': float(self.V.mean()),
            'max_V': float(self.V.max()),
            'mean_T': float(self.T.mean()),
            'sleeping': self.sleeping,
            'n_spikes': len(self.last_spikes),
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

            # 3a. STDP-LTP：当前神经元放电（作为post），
            #     对每个入向连接，检查pre是否先于post放电
            for (src, ci) in self.in_adj[idx]:
                t_pre = self.last_spike[src]
                if t_pre >= 0:
                    dt = self.step_num - t_pre
                    if 0 < dt <= self.stdp_window:
                        # pre先于post → LTP
                        delta = self.ltp_rate * np.exp(-dt / self.stdp_tau)
                        self.w[ci] = min(1.0, self.w[ci] + delta)

            # 3b. STDP-LTD：当前神经元放电（作为pre），
            #     对每个出向连接，检查post是否先于pre放电
            for (tgt, ci) in self.out_adj[idx]:
                t_post = self.last_spike[tgt]
                if t_post >= 0:
                    dt = self.step_num - t_post
                    if 0 < dt <= self.stdp_window:
                        # post先于pre → LTD
                        delta = self.ltd_rate * np.exp(-dt / self.stdp_tau)
                        self.w[ci] = max(0.0, self.w[ci] - delta)

            # 3c. 脉冲分发：按权重放入目标缓存（单步延迟）
            for (tgt, ci) in self.out_adj[idx]:
                self.spike_cache[tgt] += self.w[ci]

            # 3d. 减法重置（保留惯性）
            self.V[idx] -= self.T[idx]
            self.last_spike[idx] = self.step_num

        # 4. 动态阈值：滑动窗口平均自适应
        self.pbuf[:, self.wptr] = self.V
        self.wptr = (self.wptr + 1) % self.wsize
        self.wcount = min(self.wcount + 1, self.wsize)
        if self.wcount >= 5:
            n = self.wcount
            avg = self.pbuf[:, :n].mean(axis=1)
            self.T = np.clip(self.th_base + self.th_alpha * avg,
                             self.th_min, self.th_max)

        # 5. 睡眠态结构可塑性
        if self.sleeping:
            self._sleep_plasticity(spiking)

        # 6. 记录放电
        self.last_spikes = spiking
        self.step_num += 1
        return spiking

    # ==================== 睡眠态结构可塑性 ====================
    def _sleep_plasticity(self, spiking):
        """睡眠态：共放电统计 + 结构生长 + 剪枝"""
        # 统计共同放电
        spike_list = spiking.tolist()
        for i in range(len(spike_list)):
            for j in range(i + 1, len(spike_list)):
                a, b = spike_list[i], spike_list[j]
                key = (min(a, b), max(a, b))
                self.cofire[key] = self.cofire.get(key, 0) + 1

        # 每20步：生长+剪枝
        if self.step_num % 20 == 0 and self.step_num > 0:
            # 生长：频繁共放电且无连接的对
            for (a, b), count in list(self.cofire.items()):
                if count > 2 and len(self.out_adj[a]) < 80:
                    exists = any(t == b for t, _ in self.out_adj[a])
                    if not exists:
                        # 检查连接半径
                        d = np.linalg.norm(self.pos[a] - self.pos[b])
                        if d < self.conn_radius:
                            self._add_conn(a, b, 0.05)

            # 剪枝：权重衰减至零的连接
            for ci in np.where(self.w < 0.001)[0]:
                self._del_conn(int(ci))

            self.cofire.clear()

    def _add_conn(self, p, q, wt):
        """添加新连接（优先复用死条目，防止数组膨胀）"""
        if self._free_list:
            ci = self._free_list.pop()
            self.pre[ci] = p
            self.post[ci] = q
            self.w[ci] = wt
        else:
            ci = len(self.w)
            self.pre = np.append(self.pre, p)
            self.post = np.append(self.post, q)
            self.w = np.append(self.w, wt)
        self.out_adj[p].append((q, ci))
        self.in_adj[q].append((p, ci))
        self.nc += 1

    def _del_conn(self, ci):
        """删除连接（权重置零，回收索引，从邻接表移除）"""
        if ci >= len(self.w) or self.w[ci] == 0:
            return
        p = int(self.pre[ci])
        q = int(self.post[ci])
        self.w[ci] = 0
        self._free_list.append(ci)
        self.out_adj[p] = [(t, i) for t, i in self.out_adj[p] if i != ci]
        self.in_adj[q] = [(s, i) for s, i in self.in_adj[q] if i != ci]
        self.nc -= 1

    def enter_sleep(self):
        """进入睡眠态"""
        self.sleeping = True

    def exit_sleep(self):
        """退出睡眠态"""
        self.sleeping = False
