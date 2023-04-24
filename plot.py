import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


filename = 'output/processed_data/B03.txt'
Y = np.loadtxt(filename)
X = []

a = len(Y)

for i in np.arange(1,int((a/5000)+1), 0.0002):
    # b = [i]*5000
    X.extend([i])

X = np.array(X)
Y = np.array(Y).reshape(-1)

plt.figure(figsize=(10, 4), dpi=240)
plt.plot(X,Y)
plt.legend(['B03'], prop={'family': 'SimHei', 'size': 10})
plt.xlabel('时间（h）', fontproperties={'family': 'SimHei', 'size': 10})
plt.ylabel('振动加速度（g）', fontproperties={'family': 'SimHei', 'size': 10})
plt.show()

