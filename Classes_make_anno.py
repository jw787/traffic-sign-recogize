#! /usr/bin/env python
# coding=utf-8

import os
import pandas as pd
from PIL import Image


class Classes_anno():
    def __init__(self, root):
        self.root = root
    def make_anno(self):
        PHASE = ['train', 'test']
        KIND = [str(i).zfill(5) for i in range(62)]
        DATA_INFO = {'train':{'path': [], 'classes': []},'test':{'path': [], 'classes': []}}

        for p in PHASE:
            for k in KIND:
                # 获取和遍历图片所在的文件夹路径
                DATA_DIR = self.root + p + '/' + k
                # 列出文件夹内的文件的名字
                FILE = os.listdir(DATA_DIR)

                for file in FILE:
                    # 获取并生成文件路径
                    file_path = os.path.join(DATA_DIR, file)
                    # 这里意思是，尝试一下可不可以打看图片，如果可一打开，就pass，进行下一步，除了操作系统异常的情况下
                    try:
                        Image.open(file_path)
                    except OSError:
                        pass
                    else:
                        # 第一个括号这的是最外面键的索引，第二个括号里面的代表着嵌入字典里的键
                        DATA_INFO[p]['path'].append(file_path)
                        # 这里的可为00000，因为Python默认在转换成字符串时忽略前面的0，因此可以通过格式转换的方法实现去0：
                        #
                        # >>> str(000001)
                        # '1'
                        # >>> int(str(000001))
                        # 1
                        DATA_INFO[p]['classes'].append(int(str(k)))
            # 生成DateFrame，分别生成
            ANNOTATION = pd.DataFrame(DATA_INFO[p])
            ANNOTATION.to_csv(f'Classes_{p}_annotation.csv')
            print(f'Classes_{p}_annotation.csv is saved')

if __name__ == "__main__":
    ROOT = '../traffic_sign/'
    kind = Classes_anno(ROOT)
    kind.make_anno()
    # PHASE = ['train', 'test']
    # KIND = [str(i).zfill(5) for i in range(62)]

