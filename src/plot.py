# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os 

# --- 字体配置 (严格匹配其他分析脚本标准) ---
FONT_TITLE = 24
FONT_LABEL = 20
FONT_TICK  = 16
FONT_LEGEND = 14

# 1. 准备原始数据
data_source = {
    'Horizon': [5, 4, 3, 2, 1, 8, 10],
    'Naive Outdated': [637.977905, 548.918090, 466.693451, 379.362030, 286.660858, 898.50714, 1072.235962],
    'Transformer': [374.661804, 335.86340, 296.946838, 258.67675, 221.577347, 486.337524, 561.10992],
    'GRU': [566.235046, 441.83151, 358.341980, 283.97332, 227.149429, 968.07073, 1223.859009]
}

df = pd.DataFrame(data_source)

# 2. 数据预处理
# 按照 'Horizon' 列进行升序排序
df = df.sort_values(by='Horizon')

# 3. 转换为 dB
model_columns = ['Naive Outdated', 'Transformer', 'GRU']
df_db = df.copy()
df_db[model_columns] = 10 * np.log10(df[model_columns])

# 4. 开始绘图
# 增大 figsize 以适应大号字体
plt.figure(figsize=(20, 10))

# 绘制三条线 (颜色匹配指定要求)
# Naive -> 蓝色，Transformer -> 黄色，GRU -> 绿色
plt.plot(df_db['Horizon'], df_db['Naive Outdated'], label='Naive Outdated', 
         color='blue', marker='o', linestyle='-', linewidth=3)

plt.plot(df_db['Horizon'], df_db['Transformer'], label='Transformer', 
         color='orange', marker='s', linestyle='-', linewidth=3)

plt.plot(df_db['Horizon'], df_db['GRU'], label='GRU', 
         color='green', marker='^', linestyle='-.', linewidth=3)

# 5. 添加图表元素
plt.title('Performance Comparison: Historical Timesteps vs Prediction Horizon', 
          fontsize=FONT_TITLE, fontweight='bold', pad=20)

plt.xlabel('Horizon (timesteps)', fontsize=FONT_LABEL, fontweight='bold')
# Y轴标签加粗
plt.ylabel('Magnitude MSE (dB)', fontsize=FONT_LABEL, fontweight='bold')

# 设置轴刻度字体
plt.xticks(df_db['Horizon'], fontsize=FONT_TICK)
plt.yticks(fontsize=FONT_TICK)

# 添加图例
plt.legend(fontsize=FONT_LEGEND, loc='best', frameon=True, framealpha=0.9)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.7)

# 优化布局
plt.tight_layout()

# 6. 保存为 PDF
output_filename = 'MSE_dB_Comparison_Final.pdf'

# 创建 results 文件夹 (如果不存在)
if not os.path.exists('results'):
    os.makedirs('results')

save_path = os.path.join('results', output_filename)
plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='pdf')

print(f"高清 PDF 图像已成功保存为: {save_path}")

# 显示图像
plt.show()