"""实验通用配置：中文字体、路径"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# 中文字体
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 尝试注册Noto字体
try:
    import matplotlib.font_manager as fm
    for fontpath in [
        '/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf',
        '/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.ttf',
    ]:
        if os.path.exists(fontpath):
            fm.fontManager.addfont(fontpath)
except:
    pass

plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']

EXPERIMENT_DIR = os.path.dirname(__file__)
