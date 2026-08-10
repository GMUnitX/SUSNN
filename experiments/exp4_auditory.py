"""
实验4：听觉预测

将音频信号分解为频谱（32频段），映射到Face1的前32个神经元。
其余神经元空闲（可留作其他感官扩展）。
网络预测下一帧的频谱。

测试模式：
- 交替播放两个不同频率的纯音（类似亮度交替，但切换更快）
- 看网络能否学会预测频率切换
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common import *  # 中文字体+路径

import numpy as np
from pulse_engine import PulseEngine
from periphery import AudioEncoder, PredictionLoop

NUM_FREQ = 32
NUM_FACE = 768  # 标准面大小，前32个用于音频
NUM_HIDDEN = 100

def run():
    engine = PulseEngine(num_face=NUM_FACE, num_hidden=NUM_HIDDEN,
                         conn_radius=0.25, n_init_conn=2000, seed=42)
    audio = AudioEncoder(num_freq_bins=NUM_FREQ, num_face=NUM_FACE)
    loop = PredictionLoop(engine, NUM_FACE, steps_per_frame=1)

    TOTAL = 1200

    # 生成交替音频：每100帧切换频率（200Hz <-> 800Hz）
    def get_tone(t):
        freq = 200 if (t // 100) % 2 == 0 else 800
        # 取一小段音频
        sr = 8000
        chunk = audio.generate_tone(freq, 1/sr * 50, sr)  # 50样本
        return chunk, freq

    rec = {k: [] for k in ['t','freq','mse','spike','dominant_pred']}
    snapshots = {}

    for t in range(TOTAL):
        chunk, freq = get_tone(t)
        spectrum = audio.encode(chunk, 8000)  # 32频段

        # 放入Face1前32个神经元
        stimulus = np.zeros(NUM_FACE)
        stimulus[:NUM_FREQ] = spectrum

        prediction, error, n_spikes = loop.cycle(stimulus)

        # 只看音频通道的MSE
        pred_spectrum = prediction[:NUM_FREQ]
        mse = float(np.mean((spectrum - pred_spectrum)**2))

        # 预测的主频
        if pred_spectrum.max() > 0:
            dom_pred = float(np.argmax(pred_spectrum))
        else:
            dom_pred = -1

        rec['t'].append(t)
        rec['freq'].append(freq)
        rec['mse'].append(mse)
        rec['spike'].append(n_spikes)
        rec['dominant_pred'].append(dom_pred)

        if t in [0, 50, 200, 600, 1199]:
            snapshots[t] = (spectrum.copy(), pred_spectrum.copy(), freq)

        if t % 200 == 0:
            print(f"  {t:4d}  freq={freq}Hz  mse={mse:.6f}  dom_pred_bin={dom_pred}  spikes={n_spikes}")

    # 画图
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    for k in rec: rec[k] = np.array(rec[k])

    # 频率标记
    freqs = np.array(rec['freq'])
    freq_changes = np.where(np.diff(freqs) != 0)[0]

    axes[0].plot(rec['t'], rec['freq'], 'b-', lw=2)
    for fc in freq_changes:
        axes[0].axvline(fc, color='gray', ls='--', alpha=0.3)
    axes[0].set_ylabel('频率 (Hz)')
    axes[0].set_title('输入频率切换（200Hz <-> 800Hz）')
    axes[0].grid(alpha=0.3)

    axes[1].plot(rec['t'], rec['mse'], 'b-', lw=1, alpha=0.3)
    smooth = np.convolve(rec['mse'], np.ones(20)/20, mode='same')
    axes[1].plot(rec['t'], smooth, 'r-', lw=2, label='MSE滑动平均')
    for fc in freq_changes:
        axes[1].axvline(fc, color='gray', ls='--', alpha=0.3)
    axes[1].set_ylabel('MSE')
    axes[1].set_title('频谱预测误差')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_yscale('log')

    axes[2].plot(rec['t'], rec['spike'], 'g-', lw=1, alpha=0.5)
    smooth_s = np.convolve(rec['spike'], np.ones(20)/20, mode='same')
    axes[2].plot(rec['t'], smooth_s, 'r-', lw=2)
    axes[2].set_ylabel('放电数')
    axes[2].set_xlabel('帧')
    axes[2].set_title('网络活动')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'exp4_result.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"已保存 {out}")

    # 频谱快照
    fig, axes = plt.subplots(len(snapshots), 2, figsize=(14, 3*len(snapshots)))
    if len(snapshots) == 1:
        axes = axes.reshape(1, -1)
    for i, (t, (actual, pred, freq)) in enumerate(snapshots.items()):
        x = np.arange(NUM_FREQ)
        axes[i, 0].bar(x, actual, color='blue', alpha=0.7)
        axes[i, 0].set_title(f'帧{t} 实际频谱 (freq={freq}Hz)', fontsize=10)
        axes[i, 0].set_ylabel('能量')
        axes[i, 1].bar(x, np.clip(pred, 0, 1), color='red', alpha=0.7)
        axes[i, 1].set_title(f'帧{t} 预测频谱', fontsize=10)
    plt.suptitle('听觉预测：实际 vs 预测频谱', fontsize=14, y=1.01)
    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), 'exp4_spectra.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"已保存 {out2}")

    # 分析
    fqa = rec['mse'][:TOTAL//4].mean()
    fqb = rec['mse'][-TOTAL//4:].mean()
    print(f"\n=== 听觉预测分析 ===")
    print(f"  前段MSE: {fqa:.6f}")
    print(f"  后段MSE: {fqb:.6f}")

if __name__ == '__main__':
    run()
