import os
import numpy as np
import pandas as pd
import os.path
from nptdms import TdmsFile
import math


path = r'F:/NU218_S'

for path_1 in ['B03']:
# for path_1 in ['B02', 'B03', 'B08', 'B057']:
    bearings_type = next(os.walk(path + '/' + path_1))[1]
    bearings_type.sort()
    data = []
    length = 0
    for bearings_time in bearings_type:

        file_names = os.listdir(path + '/' + path_1 +'/' + bearings_time)
        file_names.sort()
        for i in file_names:
            if os.path.splitext(i)[1] == ".tdms":
                if 'Acceleration' in os.path.splitext(i)[0] or 'filtered_data' in os.path.splitext(i)[0]:
                    tdms_file = TdmsFile.read(path + '/' + path_1 +'/' + bearings_time+'/' +i)
                    data1 = []
                    for group in tdms_file.groups():
                            group_name = group.name
                            # print(group_name)
                            if 'All Data' in group_name:
                                for channel in group.channels():
                                    channel_name = channel.name
                                    if 'TRI-ACC' in channel_name or 'filtered data' in channel_name or 'cDAQ9184' in channel_name:
                                        print(channel_name)
                                        channel = tdms_file[group_name][channel_name]  # 根据索引读取通道
                                        all_channel_data = channel[:]
                                        # num = np.array(all_channel_data)
                                        a = len(all_channel_data)
                                        length = length + a
                                        # num = all_channel_data[0:a:18000000]
                                        for h in range(math.floor(a/18000000)):
                                            num = all_channel_data[h * 18000000: h * 18000000 + 5000]
                                            data1.extend(num)


                    data.extend(data1)
                    break
    print(path_1)
    print(length)


    np.savetxt(r"output/processed_data/" + str(path_1) + '.txt', data, delimiter=',')