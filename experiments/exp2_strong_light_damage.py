"""
实验2：强光损伤与恢复

阶段1: 正常亮度适应（200帧）→ 建立基线
阶段2: 持续强光暴露（500帧，亮度=3.0）→ 模拟强光损伤
阶段3: 恢复期，正常亮度适应（300帧）→ 看适应能力是否恢复

假设：长时间强光暴露导致阈值被推得极高（动态阈值自适应），
     撤掉强光后阈值不会立即回落 → 适应能力暂时退化。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common import *  # 中文字体+路径

import numpy as np
from pulse_engine import PulseEngine
from periphery import ActionNeuron

NUM_PIXELS = 16 * 16 * 3
NUM_FACE = NUM_PIXELS + 1
NUM_HIDDEN = 100
CTRL_IDX = NUM_FACE + NUM_PIXELS

def get_brightness(t, phase_info):
    """亮度调度：正常→强光→恢复"""
    phase, t_in = phase_info
    if phase == 'normal':
        return 1.0 if (t_in // 100) % 2 == 0 else 0.1
    elif phase == 'damage':
        return 3.0  # 持续强光
    elif phase == 'recovery':
        return 1.0 if (t_in // 100) % 2 == 0 else 0.1

def run():
    engine = PulseEngine(num_face=NUM_FACE, num_hidden=NUM_HIDDEN,
                         conn_radius=0.25, n_init_conn=2000, seed=42)
    action = ActionNeuron(engine, CTRL_IDX, ActionNeuron.brightness_gain)

    prev_control = 0.0
    prev_face2 = np.zeros(NUM_FACE)
    rec = {k: [] for k in ['t','phase','bright','ctrl','gain','adj','err','conn','spike','mean_T']}

    phases = [('normal', 200), ('damage', 500), ('recovery', 300)]
    t = 0
    for phase_name, duration in phases:
        for t_in in range(duration):
            bright = get_brightness(t_in, (phase_name, t_in))
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
            rec['phase'].append(phase_name)
            rec['bright'].append(bright)
            rec['ctrl'].append(control)
            rec['gain'].append(gain)
            rec['adj'].append(adjusted)
            rec['err'].append(adjusted - control)
            rec['conn'].append(engine.nc)
            rec['spike'].append(len(spikes))
            rec['mean_T'].append(engine.T.mean())

            if t % 100 == 0 or t_in == duration - 1:
                print(f"  {t:4d} [{phase_name:8s}] bright={bright:.1f} ctrl={control:.4f} gain={gain:.3f} T={engine.T.mean():.3f} spikes={len(spikes)}")
            t += 1

    # 画图
    fig, axes = plt.subplots(5, 1, figsize=(18, 20), sharex=True)
    for k in rec:
        if k != 'phase': rec[k] = np.array(rec[k])

    # 背景色带
    boundaries = [0, 200, 700, 1000]
    colors = ['lightyellow', 'lightcoral', 'lightyellow']
    labels = ['正常', '强光损伤', '恢复']
    for ax in axes:
        for i in range(3):
            ax.axvspan(boundaries[i], boundaries[i+1], alpha=0.15, color=colors[i])
            if ax == axes[0]:
                ax.text(boundaries[i]+10, ax.get_ylim()[1]*0.9 if ax.get_ylim()[1] > 0 else 1, labels[i], fontsize=12)

    axes[0].plot(rec['t'], rec['bright'], 'b-', lw=2, label='原始亮度')
    axes[0].plot(rec['t'], rec['adj'], 'r-', lw=2, label='调整后')
    axes[0].set_ylabel('亮度'); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title('强光损伤实验：亮度')

    axes[1].plot(rec['t'], rec['ctrl'], 'g-', lw=2, label='控制信号')
    axes[1].set_ylabel('控制'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title('Face2控制信号')

    axes[2].plot(rec['t'], rec['gain'], 'm-', lw=2, label='增益')
    axes[2].axhline(1.0, color='gray', ls='--', alpha=0.5)
    axes[2].set_ylabel('增益'); axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].set_title('增益系数')

    axes[3].plot(rec['t'], rec['mean_T'], 'k-', lw=2, label='平均阈值')
    axes[3].set_ylabel('阈值'); axes[3].legend(); axes[3].grid(alpha=0.3)
    axes[3].set_title('动态阈值（强光推高→恢复期回落？）')

    axes[4].plot(rec['t'], np.abs(rec['err']), 'k-', lw=1, alpha=0.5)
    smooth = np.convolve(np.abs(rec['err']), np.ones(10)/10, mode='same')
    axes[4].plot(rec['t'], smooth, 'r-', lw=2, label='|误差|滑动平均')
    axes[4].set_ylabel('|误差|'); axes[4].legend(); axes[4].grid(alpha=0.3)
    axes[4].set_title('适应误差（恢复期是否回到正常水平？）')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'exp2_result.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n已保存 {out}")

    # 分析
    print("\n=== 强光损伤分析 ===")
    normal_end = 200
    damage_end = 700
    for label, start, end in [('正常末', 150, 200), ('强光末', 650, 700), ('恢复初', 700, 750), ('恢复末', 950, 1000)]:
        mask = (rec['t'] >= start) & (rec['t'] < end)
        if mask.any():
            print(f"  {label}: 阈值={rec['mean_T'][mask].mean():.3f} ctrl={rec['ctrl'][mask].mean():.4f} "
                  f"|误差|={np.abs(rec['err'][mask]).mean():.4f}")

if __name__ == '__main__':
    run()
