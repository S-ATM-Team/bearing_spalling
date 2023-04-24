import os
import numpy as np
import pandas as pd
import os.path
from nptdms import TdmsFile


path = r'F:/NU218_Spalling'

for path_1 in ['/1010', '/B02', '/B03', '/B05', '/B08']:
    bearings_type = next(os.walk(path + path_1))[1]
    bearings_type.sort()
    data = []
    for bearings_time in bearings_type:

        file_names = os.listdir(path + path_1 +'/' + bearings_time)
        file_names.sort()
        for i in file_names:
            if os.path.splitext(i)[1] == ".tdms":
                if 'Acceleration' in os.path.splitext(i)[0] or 'filtered_data' in os.path.splitext(i)[0]:
                    tdms_file = TdmsFile.read(path + path_1 +'/' + bearings_time+'/' +i)
                    data1 = []
                    for group in tdms_file.groups():
                            group_name = group.name
                            print(group_name)
                            if 'All Data' in group_name:
                                for channel in group.channels():
                                    channel_name = channel.name
                                    print(channel_name)
                                    channel = tdms_file[group_name][channel_name]  # 根据索引读取通道
                                    all_channel_data = channel[:]
                                    num = np.array(all_channel_data)
                                    data1.append(num)
                    data.extend(data1[0])
                    break

    np.savetxt(r"output/processed_data/" + str(path_1) + '.txt', data, delimiter=',')




