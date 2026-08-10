"""
实验3：运动预测

一条亮条在画面上从左到右匀速移动，环绕循环。
网络通过Face1接收当前帧，Face2输出预测。
误差 = 下一帧实际 - 当前预测。
看网络能否学会预测运动方向。

同时跟踪：
- 像素级预测误差（MSE）随时间下降
- 预测的"提前量"（预测是否领先于实际）
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common import *  # 中文字体+路径

import numpy as np
from pulse_engine import PulseEngine
from periphery import VisualEncoder, PredictionLoop

IMG_W = 16
IMG_H = 16
NUM_PIXELS = IMG_W * IMG_H * 3
NUM_FACE = NUM_PIXELS
NUM_HIDDEN = 100

def run():
    engine = PulseEngine(num_face=NUM_FACE, num_hidden=NUM_HIDDEN,
                         conn_radius=0.25, n_init_conn=2000, seed=42)
    vis = VisualEncoder(IMG_W, IMG_H)
    loop = PredictionLoop(engine, NUM_FACE, steps_per_frame=1)

    TOTAL = 1200
    BAR_SPEED = 1.0  # 每帧移动1像素

    rec = {k: [] for k in ['t','mse_echo','mse_pred','spike','conn','bar_x']}
    snapshots = {}  # 关键帧的预测可视化

    prev_prediction = None
    prev_img = None

    for t in range(TOTAL):
        bar_x = (t * BAR_SPEED) % IMG_W
        img = vis.moving_bar(bar_x, bar_width=2, brightness=0.8)
        current = vis.encode(img)

        prediction, error, n_spikes = loop.cycle(current)

        # 回声误差：预测 vs 当前帧（网络是否在回声输入）
        mse_echo = float(np.mean((current - prediction[:NUM_PIXELS])**2))

        # 真预测误差：上一帧的预测 vs 当前帧（网络是否在预测下一帧）
        if prev_prediction is not None:
            mse_pred = float(np.mean((current - prev_prediction[:NUM_PIXELS])**2))
        else:
            mse_pred = 0.0

        rec['t'].append(t)
        rec['mse_echo'].append(mse_echo)
        rec['mse_pred'].append(mse_pred)
        rec['spike'].append(n_spikes)
        rec['conn'].append(engine.nc)
        rec['bar_x'].append(bar_x)

        # 快照：展示预测 vs 下一帧实际
        if t in [50, 200, 600, 1199]:
            pred_img = vis.decode_prediction(prediction)
            snapshots[t] = (img.copy(), pred_img.copy(), mse_pred)

        prev_prediction = prediction.copy()
        prev_img = img.copy()

        if t % 200 == 0:
            print(f"  {t:4d}  bar_x={bar_x:.1f}  echo={mse_echo:.6f}  pred={mse_pred:.6f}  spikes={n_spikes}")

    # 画图
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    for k in rec: rec[k] = np.array(rec[k])

    axes[0].plot(rec['t'], rec['mse_echo'], 'b-', lw=1, alpha=0.3, label='回声误差(预测vs当前)')
    axes[0].plot(rec['t'], rec['mse_pred'], 'g-', lw=1, alpha=0.3, label='真预测误差(上帧预测vs当前)')
    smooth = np.convolve(rec['mse_pred'], np.ones(20)/20, mode='same')
    axes[0].plot(rec['t'], smooth, 'r-', lw=2, label='真预测误差滑动平均')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('运动预测误差（真预测误差下降=学会预测运动）')
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_yscale('log')

    axes[1].plot(rec['t'], rec['spike'], 'g-', lw=1, alpha=0.5)
    smooth_s = np.convolve(rec['spike'], np.ones(20)/20, mode='same')
    axes[1].plot(rec['t'], smooth_s, 'r-', lw=2, label='放电数滑动平均')
    axes[1].set_ylabel('放电数')
    axes[1].set_xlabel('帧')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title('网络活动水平')

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'exp3_result.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"已保存 {out}")

    # 快照图
    fig, axes = plt.subplots(len(snapshots), 2, figsize=(12, 4*len(snapshots)))
    if len(snapshots) == 1:
        axes = axes.reshape(1, -1)
    for i, (t, (actual, pred, mse)) in enumerate(snapshots.items()):
        axes[i, 0].imshow(actual)
        axes[i, 0].set_title(f'帧{t} 实际', fontsize=10)
        axes[i, 0].axis('off')
        axes[i, 1].imshow(np.clip(pred, 0, 1))
        axes[i, 1].set_title(f'帧{t} 预测 (MSE={mse:.4f})', fontsize=10)
        axes[i, 1].axis('off')
    plt.suptitle('运动预测：实际 vs 预测', fontsize=14, y=1.01)
    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), 'exp3_snapshots.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"已保存 {out2}")

    # 分析
    first_echo = rec['mse_echo'][:TOTAL//4].mean()
    last_echo = rec['mse_echo'][-TOTAL//4:].mean()
    first_pred = rec['mse_pred'][10:TOTAL//4].mean()  # skip first few
    last_pred = rec['mse_pred'][-TOTAL//4:].mean()
    print(f"\n=== 运动预测分析 ===")
    print(f"  回声误差: 前={first_echo:.6f} 后={last_echo:.6f}")
    print(f"  真预测误差: 前={first_pred:.6f} 后={last_pred:.6f}")
    if first_pred > 0:
        print(f"  真预测下降: {(1-last_pred/first_pred)*100:.1f}%")

if __name__ == '__main__':
    run()
