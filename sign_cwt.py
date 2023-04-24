import pywt
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd

def prepro(d_path, length, number, normal, rate):
    filenames = os.listdir(d_path)
    def capture(d_path):
        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            files[i] = np.array(pd.read_csv(file_path, header=None)).reshape(-1)
        return files

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            samp_train = int(number * (1 - slice_rate))  # 1000(1-0.3)
            Train_sample = []
            Test_Sample = []

            for j in range(samp_train):
                # sample = slice_data[j*10000: j*10000 + length]
                sample = slice_data[1280000+j * 4000: 1280000+j * 4000 + length]
                Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                # sample = slice_data[samp_train*10000 + length + h*10000: samp_train*10000 + length + h*10000 + length]
                sample = slice_data[1280000+samp_train * 4000 + length + h * 4000: 1280000+samp_train * 4000 + length + h * 4000 + length]
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

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        data_all = np.vstack((Train_X, Test_X))
        scalar = preprocessing.StandardScaler().fit(data_all)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):

        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)

        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]

            return X_valid, Y_valid, X_test, Y_test

    data = capture(d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)

    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    Train_X = np.asarray(Train_X)
    Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


path = r'data/L10V200'
# path = r'F:/NU218_Artificial_processed/10KN/B011'
x_train, y_train, x_valid, y_valid, x_test, y_test = prepro(d_path=path,length=5120,number=70,normal=True,rate=[0.6, 0.2, 0.2])

for i in range(0, len(x_valid)):
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
    [cwtmatr, frequencies] = pywt.cwt(x_valid[i], scales, wavename, 1.0 / fs)
    plt.contourf(t, frequencies, abs(cwtmatr))

    plt.axis('off')
    plt.gcf().set_size_inches(5120 / 1000, 5120 / 1000)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    x = r'cwt_L10V200/test/' + str(i) + '-' + str(y_valid[i]) + '.jpg'
    plt.savefig(x)