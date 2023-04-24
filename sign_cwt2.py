import pywt
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
import pandas as pd


def prepro(d_path, length, number, normal, rate ):
    filenames = os.listdir(d_path)
    def capture(original_path):
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            files[i] = np.array(pd.read_csv(file_path, header=None)).reshape(-1)
        return files

    def slice_enc(data, slice_rate=rate[1]):
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            samp_train = int(number * (1 - slice_rate))
            Train_sample = []
            Test_Sample = []
            for j in range(samp_train):
                sample = slice_data[1280000+j*4000: 1280000+j*4000 + length]
                Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                sample = slice_data[1280000+samp_train*4000 + length + h*4000: 1280000+samp_train*4000 + length + h*4000 + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    def scalar_stand(Train_X):
        # 用训练集标准差标准化训练集以及测试集
        # data_all = np.vstack((Train_X, Test_X))
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        # Test_X = scalar.transform(Test_X)
        return Train_X

    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)

    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X = scalar_stand(Train_X)
    Train_X = np.asarray(Train_X)
    return Train_X, Train_Y

path = r'data/L10V400'
# path = r'F:/NU218_Artificial_processed/10KN/B012'
# path = r'F:/NU218_Artificial_processed/10KN/B011'
x_train, y_train = prepro(d_path=path,length=5120,number=70,normal=True,rate=[1, 0])

for i in range(650, len(x_train)):
    # N = 784
    N = 5120
    fs = 51200#采样频率
    # 采样数据的时间维度
    t = np.linspace(0, 5120 / fs, N, endpoint=False)
    wavename = 'cmor3-3'
    totalscal = 256
    # 中心频率
    fc = pywt.central_frequency(wavename)#小波的中心频率
    # 计算对应频率的小波尺度
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(x_train[i], scales, wavename, 1.0 / fs)
    plt.contourf(t, frequencies, abs(cwtmatr))

    plt.axis('off')
    plt.gcf().set_size_inches(5120 / 1000, 5120/ 1000)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    x = r'Spalling_area/32/all/' + str(i) + '-' + str(y_train[i]) + '.jpg'
    plt.savefig(x)