"""
外围交互回路（引擎外部应用层）

包含：
1. 感官编码：视觉/听觉/触觉 → Face1 信号
2. 预测误差闭环：输入→预测→误差→反馈
3. 动作执行扩展：读取动作神经元 → 控制外部环境
"""

import numpy as np


# ==================== 感官编码 ====================

class VisualEncoder:
    """视觉编码：RGB图像 → Face1信号"""

    def __init__(self, img_width=16, img_height=16):
        self.w = img_width
        self.h = img_height
        self.n = img_width * img_height * 3  # RGB

    def encode(self, image):
        """将 (H,W,3) 图像编码为 (n,) 信号"""
        return image.flatten().astype(np.float64)

    def decode_prediction(self, face2_output):
        """将Face2输出解码为 (H,W,3) 图像"""
        return face2_output[:self.n].reshape(self.h, self.w, 3)

    def uniform_image(self, brightness, color=(1, 1, 1)):
        """生成均匀亮度的图像"""
        img = np.zeros((self.h, self.w, 3))
        for c in range(3):
            img[:, :, c] = brightness * color[c]
        return img

    def moving_bar(self, bar_x, bar_width=2, brightness=1.0):
        """生成移动光条图像"""
        img = np.zeros((self.h, self.w, 3))
        x0 = int(bar_x) % self.w
        for dw in range(bar_width):
            x = (x0 + dw) % self.w
            img[:, x, :] = brightness
        return img


class AudioEncoder:
    """听觉编码：音频频谱 → Face1信号

    将音频信号分解为N个频段的能量，映射到Face1的一部分神经元。
    """

    def __init__(self, num_freq_bins=32, num_face=769):
        self.n_bins = num_freq_bins
        self.offset = 0  # 音频通道从Face1第0个神经元开始
        # 其余神经元可以留给其他感官

    def encode(self, audio_signal, sample_rate=8000):
        """将音频信号编码为频谱信号"""
        # 确保信号足够长以产生所需频率bin数
        min_len = 2 * self.n_bins
        if len(audio_signal) < min_len:
            audio_signal = np.pad(audio_signal, (0, min_len - len(audio_signal)))
        fft = np.fft.rfft(audio_signal)
        mag = np.abs(fft[:self.n_bins])
        max_mag = mag.max() + 1e-8
        return mag / max_mag

    def generate_tone(self, freq, duration, sample_rate=8000):
        """生成单音"""
        t = np.arange(int(sample_rate * duration)) / sample_rate
        return np.sin(2 * np.pi * freq * t)

    def generate_chord(self, freqs, duration, sample_rate=8000):
        """生成和弦"""
        t = np.arange(int(sample_rate * duration)) / sample_rate
        signal = np.zeros_like(t)
        for f in freqs:
            signal += np.sin(2 * np.pi * f * t)
        return signal / len(freqs)

    def generate_sweep(self, f0, f1, duration, sample_rate=8000):
        """生成频率扫描"""
        t = np.arange(int(sample_rate * duration)) / sample_rate
        freq = f0 + (f1 - f0) * t / duration
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        return np.sin(phase)


class TactileEncoder:
    """触觉编码：压力阵列 → Face1信号"""

    def __init__(self, grid_size=16, num_face=None):
        self.gs = grid_size
        self.n = grid_size * grid_size
        self.num_face = num_face

    def encode(self, pressure_grid):
        """将 (gs,gs) 压力阵列编码为信号"""
        return pressure_grid.flatten().astype(np.float64)

    def moving_touch(self, x, y, radius=2, pressure=1.0):
        """生成移动触压点图像"""
        grid = np.zeros((self.gs, self.gs))
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                px = int(x + dx) % self.gs
                py = int(y + dy) % self.gs
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= radius:
                    grid[py, px] = pressure * (1 - dist / radius)
        return grid

    def tapping_pattern(self, step, locations=None):
        """生成敲击模式"""
        if locations is None:
            locations = [(4, 4), (8, 8), (12, 4), (4, 12)]
        grid = np.zeros((self.gs, self.gs))
        idx = step % len(locations)
        if step % 2 == 0:  # 每隔一帧敲击一次
            x, y = locations[idx]
            grid[y, x] = 1.0
            grid[y, x-1] = grid[y, x+1] = 0.5
        return grid


# ==================== 预测误差闭环 ====================

class PredictionLoop:
    """预测误差闭环控制器

    按固定采样周期运行：
    1. 将当前感官输入注入Face1
    2. 让引擎运行若干步
    3. 从Face2读取预测
    4. 计算误差 = 下一帧真实 - 预测
    5. 将误差注入Face2
    6. 进入下一轮
    """

    def __init__(self, engine, face_size, steps_per_frame=1):
        self.engine = engine
        self.face_size = face_size
        self.steps_per_frame = steps_per_frame
        self.prev_prediction = np.zeros(face_size)
        self.prev_input = np.zeros(face_size)

    def cycle(self, current_input):
        """执行一轮预测闭环

        参数:
            current_input: 当前帧的感官输入 (face_size,)

        返回:
            prediction: Face2输出（对下一帧的预测）
            error: 误差信号（当前输入 - 上次预测）
        """
        # 注入当前输入到Face1
        self.engine.input_face_1(current_input)

        # 计算误差 = 当前输入 - 上次预测
        error = np.clip(current_input - self.prev_prediction, -1.0, 1.0)
        self.engine.input_face_2(error)

        # 运行引擎
        total_spikes = []
        for _ in range(self.steps_per_frame):
            spikes = self.engine.step()
            total_spikes.append(len(spikes))

        # 读取Face2输出作为预测
        prediction = self.engine.get_output_face_2()

        # 更新记忆
        self.prev_input = current_input.copy()
        self.prev_prediction = prediction.copy()

        return prediction, error, sum(total_spikes)


# ==================== 动作执行扩展 ====================

class ActionNeuron:
    """动作神经元：读取膜电位 → 映射为控制信号"""

    def __init__(self, engine, neuron_idx, gain_func=None):
        self.engine = engine
        self.idx = neuron_idx
        self.gain_func = gain_func or (lambda x: x)

    def read(self):
        """读取动作神经元当前值"""
        return self.engine.get_potential(self.idx)

    def read_mapped(self):
        """读取并映射为控制信号"""
        return self.gain_func(self.read())

    @staticmethod
    def brightness_gain(control):
        """亮度控制增益：高控制→低增益（调暗），低控制→高增益（调亮）"""
        return np.clip(1.5 - control, 0.1, 3.0)
