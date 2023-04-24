import os
import numpy as np
import pandas as pd
import os.path
from nptdms import TdmsFile
import math


path = r'F:/NU218_Artificial_1115'
filenameList = os.listdir(path)
for path_1 in filenameList:
    if 'B010-L10' in os.path.splitext(path_1)[0]:
        file_names = os.listdir(path + '/' + path_1)

        file_names.sort()
        for i in file_names:
            if os.path.splitext(i)[1] == ".tdms":
                if 'Acceleration' in os.path.splitext(i)[0] or 'filtered_data' in os.path.splitext(i)[0]:
                    tdms_file = TdmsFile.read(path + '/' + path_1 + '/' + i)
                    data = []
                    for group in tdms_file.groups():
                        group_name = group.name
                        # print(group_name)
                        if 'All Data' in group_name:
                            for channel in group.channels():
                                channel_name = channel.name
                                # if 'TRI-ACC' in channel_name or 'filtered data' in channel_name or 'cDAQ9184' in channel_name:
                                print(channel_name)
                                channel = tdms_file[group_name][channel_name]  # 根据索引读取通道
                                all_channel_data = channel[:]
                                # num = np.array(all_channel_data)
                                num = np.array(all_channel_data)
                                # num = num.reshape(-1, 1)
                                data.append(num)
                                # data = list(np.array(data).ravel())
                                data = np.array(data).reshape(-1,1)

                                # data = pd.DataFrame(data)

                                # data.to_csv("output/processed_data/" + str(path_1) + '.csv', mode='a', header=False, index=False)
                                np.savetxt("F:/NU218_Artificial_processed/processed_data/B010-L10/" + str(path_1) + '.csv', data, delimiter=",")

                            # data.extend(data1[0])
                            break

                        # np.savetxt(r"output/processed_data/" + str(path_1) + '.txt', data, delimiter=',')
