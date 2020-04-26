# 交通标志识别

1.第一步：整理数据集

这些代码注释在image_rename.py 和 Classes_make_anno.py都有详细的备注


2.利用迁移学习来fine-tune网络参数,详细的备注解释在Classes_classification.py都有，这里用的backbone是Resnet-18，主要是因为数据集较小，用深层的网络容易过拟合，用加权的CrossEntroyloss来作为损失函数，正确率能打到85%左右。
