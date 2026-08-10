"""
实验1：亮度自适应（瞳孔反射）

Face1额外通道传入平均亮度，Face2额外通道输出控制信号。
控制信号调节进光增益 → 亮光调暗、暗光调亮。
误差 = 实际亮度 - 控制信号（预测编码逻辑不变）。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common import *  # 中文字体+路径

import numpy as np
from pulse_engine import PulseEngine
from periphery import ActionNeuron

NUM_PIXELS = 16 * 16 * 3  # 768
NUM_FACE = NUM_PIXELS + 1  # 769
NUM_HIDDEN = 100
CONTROL_NEURON_IDX = NUM_FACE + NUM_PIXELS  # Face2最后一个

def get_brightness(t):
    return 1.0 if (t // 100) % 2 == 0 else 0.1

def run():
    engine = PulseEngine(num_face=NUM_FACE, num_hidden=NUM_HIDDEN,
                         conn_radius=0.25, n_init_conn=2000, seed=42)
    action = ActionNeuron(engine, CONTROL_NEURON_IDX,
                          ActionNeuron.brightness_gain)

    prev_control = 0.0
    prev_face2 = np.zeros(NUM_FACE)
    rec = {k: [] for k in ['t','bright','ctrl','gain','adj','err','conn','spike']}

    TOTAL = 800
    for t in range(TOTAL):
        bright = get_brightness(t)
        gain = action.gain_func(prev_control)
        adjusted = bright * gain

        pixels = np.full(NUM_PIXELS, adjusted)
        stimulus = np.zeros(NUM_FACE)
        stimulus[:NUM_PIXELS] = pixels
        stimulus[NUM_PIXELS] = adjusted
        engine.input_face_1(stimulus)

        if t > 0:
            err = np.zeros(NUM_FACE)
            err[:NUM_PIXELS] = np.clip(pixels - prev_face2[:NUM_PIXELS], -1, 1)
            err[NUM_PIXELS] = np.clip(adjusted - prev_control, -1, 1)
            engine.input_face_2(err)

        spikes = engine.step()
        face2 = engine.get_output_face_2()
        control = face2[NUM_PIXELS]
        prev_control = control
        prev_face2 = face2.copy()

        rec['t'].append(t)
        rec['bright'].append(bright)
        rec['ctrl'].append(control)
        rec['gain'].append(gain)
        rec['adj'].append(adjusted)
        rec['err'].append(adjusted - control)
        rec['conn'].append(engine.nc)
        rec['spike'].append(len(spikes))

        if t % 100 == 0:
            print(f"  {t:4d}  bright={bright:.2f}  ctrl={control:.4f}  gain={gain:.3f}  adj={adjusted:.4f}")

    # 画图
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    for k in rec: rec[k] = np.array(rec[k])

    axes[0].plot(rec['t'], rec['bright'], 'b-', lw=2, label='原始亮度')
    axes[0].plot(rec['t'], rec['adj'], 'r-', lw=2, label='调整后')
    axes[0].set_ylabel('亮度'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title('亮度自适应：原始 vs 调整后')

    axes[1].plot(rec['t'], rec['ctrl'], 'g-', lw=2, label='控制信号')
    axes[1].plot(rec['t'], rec['adj'], 'r--', lw=1, alpha=0.4, label='调整后亮度')
    axes[1].set_ylabel('控制'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title('Face2控制信号输出')

    axes[2].plot(rec['t'], rec['gain'], 'm-', lw=2, label='增益')
    axes[2].axhline(1.0, color='gray', ls='--', alpha=0.5)
    axes[2].set_ylabel('增益'); axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_title('增益系数（<1=调暗, >1=调亮）')

    axes[3].plot(rec['t'], np.abs(rec['err']), 'k-', lw=1, alpha=0.5)
    smooth = np.convolve(np.abs(rec['err']), np.ones(10)/10, mode='same')
    axes[3].plot(rec['t'], smooth, 'r-', lw=2, label='|误差|滑动平均')
    axes[3].set_ylabel('|误差|'); axes[3].legend(); axes[3].grid(alpha=0.3)
    axes[3].set_title('亮度控制通道误差')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'exp1_result.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"已保存 {out}")

    # 统计
    bright_mask = rec['bright'] > 0.5
    dark_mask = rec['bright'] < 0.5
    print(f"\n亮光阶段: ctrl={rec['ctrl'][bright_mask].mean():.4f} gain={rec['gain'][bright_mask].mean():.3f}")
    print(f"暗光阶段: ctrl={rec['ctrl'][dark_mask].mean():.4f} gain={rec['gain'][dark_mask].mean():.3f}")
    print(f"理论均衡: 亮=0.75(gain=0.75) 暗=0.1364(gain=1.364)")

if __name__ == '__main__':
    run()
