import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同
import pandas as pd
import csv
import matplotlib.pyplot as plt


path = r'data/L10V300'
filenames = os.listdir(path)

def capture(path):
    files = {}
    for i in filenames:

        file_path = os.path.join(path, i)

        files[i] = np.array(pd.read_csv(file_path, header=None)).reshape(-1)

    return files


data = capture(path)

plt.figure(figsize=(15, 15), dpi=80)

plt.subplot(10,1,1)
plt.plot(data['2.csv'], label='0')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 8})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 10})
# plt.title('100rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,2)
plt.plot(data['4.csv'],label='1')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
# plt.title('150rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,3)
plt.plot(data['8.csv'], label='2')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
# plt.title('200rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,4)
plt.plot(data['12.csv'], label='3')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
# plt.title('250rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,5)
plt.plot(data['16.csv'], label='4')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
# plt.title('300rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,6)
plt.plot(data['20.csv'], label='5')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 20})
# plt.title('400rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,7)
plt.plot(data['24.csv'], label='6')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
# plt.title('500rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,8)
plt.plot(data['26.csv'], label='7')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
# plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 15})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
# plt.title('600rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,9)
plt.plot(data['28.csv'], label='8')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 10})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
# plt.title('700rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.subplot(10,1,10)
plt.plot(data['30.csv'], label='9')
plt.legend(prop={'family':'SimHei','size':10},loc='upper left')
plt.xlabel('采样点', fontproperties={'family': 'SimHei', 'size': 20})
# plt.ylabel('振动加速度(g)', fontproperties={'family': 'SimHei', 'size': 8})
# plt.title('800rpm', fontproperties={'family': 'SimHei', 'size': 10})

plt.tight_layout()
plt.show()