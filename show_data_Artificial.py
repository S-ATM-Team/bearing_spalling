import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import pandas as pd
import csv
import matplotlib.pyplot as plt


path = r'F:/NU218_Artificial_processed/10KN/B011'
filenames = os.listdir(path)

def capture(path):
    files = {}
    for i in filenames:

        file_path = os.path.join(path, i)

        files[i] = np.array(pd.read_csv(file_path, header=None)).reshape(-1)

    return files


data = capture(path)

plt.figure(figsize=(15, 15), dpi=80)

plt.subplot(521)
plt.plot(data['V100.csv'])
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 8})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('100rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(522)
plt.plot(data['V150.csv'])
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('150rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(523)
plt.plot(data['V200.csv'])
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('200rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(524)
plt.plot(data['V250.csv'])
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('250rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(525)
plt.plot(data['V300.csv'])
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('300rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(526)
plt.plot(data['V400.csv'])
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('400rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(527)
plt.plot(data['V500.csv'])
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('500rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(528)
plt.plot(data['V600.csv'])
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('600rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(529)
plt.plot(data['V700.csv'])
plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 10})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('700rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(5, 2, 10)
plt.plot(data['V800.csv'])
plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 10})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
plt.title('800rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.tight_layout()
plt.show()