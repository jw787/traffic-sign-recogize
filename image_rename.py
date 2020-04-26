#! /usr/bin/env python
# coding=utf-8

import os
import numpy as np

PHASE = ['train', 'test']
# 将其强制转化为整形，避免下面函数调用时出现浮点数的形式
kind = np.linspace(0, 61, 62, dtype=np.int)
# 将其转化为['00000','00001'...]的形式，填补零
KIND = list(map(lambda x: str(x).zfill(5), kind))

class Image_rename():
    def __init__(self, r, s):
        self.path = r + '/' + s
    def rename(self):

        # 列出所有的路径下的文件或者文件夹名
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        n = 0
        for i in filelist:
            # 判断字符串i是不是以字符串'.png结尾.
            if i.endswith('.png'):
                # os.path.abspath(self.path): 返回一个目录的绝对路径
                oldname = os.path.join(os.path.abspath(self.path), i)
                # 这里的str.zfill是能够按照缺失的位数来补0，例如00001,00233等
                newname = os.path.join(os.path.abspath(self.path), str(k).zfill(5) + "_" + str(n).zfill(3) + '.png')
                # 将旧名替换成新名
                os.rename(oldname, newname)

                n += 1

        print(f'Data_dir{filelist} total {total_num} to rename & converted {n} pngs')



if __name__ == "__main__":
    for p in PHASE:
        for k in KIND:
            newname = Image_rename(p, k)
            newname.rename()

